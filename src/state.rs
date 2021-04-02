//! Graphics state.

use glow::HasContext;
use luminance::{
    blending::{Equation, Factor},
    depth_test::{DepthComparison, DepthWrite},
    face_culling::{FaceCullingMode, FaceCullingOrder},
    scissor::ScissorRegion,
};
use std::{fmt, marker::PhantomData};

#[derive(Debug)]
pub(crate) struct BindingStack {
    pub(crate) next_texture_unit: u32,
    pub(crate) free_texture_units: Vec<u32>,
    pub(crate) next_buffer_binding: u32,
    pub(crate) free_buffer_bindings: Vec<u32>,
}

impl BindingStack {
    // Create a new, empty binding stack.
    fn new() -> Self {
        BindingStack {
            next_texture_unit: 0,
            free_texture_units: Vec::new(),
            next_buffer_binding: 0,
            free_buffer_bindings: Vec::new(),
        }
    }
}

/// The graphics state.
///
/// This type represents the current state of a given graphics context. It acts
/// as a forward-gate to all the exposed features from the low-level API but
/// adds a small cache layer over it to prevent from issuing the same API call (with
/// the same parameters).
#[derive(Debug)]
pub struct GlowState {
    _phantom: PhantomData<*const ()>, // !Send and !Sync

    // WebGL context
    pub(crate) ctx: glow::Context,

    // binding stack
    binding_stack: BindingStack,

    // viewport
    viewport: [i32; 4],

    // clear buffers
    clear_color: [f32; 4],

    // blending
    blending_state: BlendingState,
    blending_equations: BlendingEquations,
    blending_funcs: BlendingFactors,

    // depth test
    depth_test: DepthTest,
    depth_test_comparison: DepthComparison,

    // depth write
    depth_write: DepthWrite,

    // face culling
    face_culling_state: FaceCullingState,
    face_culling_order: FaceCullingOrder,
    face_culling_mode: FaceCullingMode,

    // scissor
    scissor_state: ScissorState,
    scissor_region: ScissorRegion,

    // texture
    current_texture_unit: u32,
    bound_textures: Vec<(u32, Option<glow::Texture>)>,

    // texture buffer used to optimize texture creation; regular textures typically will never ask
    // for fetching from this set but framebuffers, who often generate several textures, might use
    // this opportunity to get N textures (color, depth and stencil) at once, in a single CPU / GPU
    // roundtrip
    //
    // fishy fishy
    texture_swimming_pool: Vec<Option<glow::Texture>>,

    // uniform buffer
    bound_uniform_buffers: Vec<Option<glow::Buffer>>,

    // array buffer
    bound_array_buffer: Option<glow::Buffer>,
    // element buffer
    bound_element_array_buffer: Option<glow::Buffer>,

    // framebuffer
    bound_draw_framebuffer: Option<glow::Framebuffer>,
    bound_read_framebuffer: Option<glow::Framebuffer>,

    // A special framebuffer used to read textures (workaround the fact WebGL2 doesn’t have
    // support of glGetTexImage). That object will never be created until trying to read a
    // texture’s image.
    readback_framebuffer: Option<glow::Framebuffer>,

    // vertex array
    bound_vertex_array: Option<glow::VertexArray>,
    // shader program
    current_program: Option<glow::Program>,
}

impl GlowState {
    /// Create a new `GLState`.
    ///
    /// > Note: keep in mind you can create only one per thread. However, if you’re building without
    /// > standard library, this function will always return successfully. You have to take extra care
    /// > in this case.
    pub(crate) fn new(ctx: glow::Context) -> Result<Self, StateQueryError> {
        Self::get_from_context(ctx)
    }

    /// Get a `GraphicsContext` from the current OpenGL context.
    fn get_from_context(mut ctx: glow::Context) -> Result<Self, StateQueryError> {
        let binding_stack = BindingStack::new();
        let viewport = get_ctx_viewport(&mut ctx)?;
        let clear_color = get_ctx_clear_color(&mut ctx)?;
        let blending_state = get_ctx_blending_state(&mut ctx);
        let blending_equations = get_ctx_blending_equations(&mut ctx)?;
        let blending_funcs = get_ctx_blending_factors(&mut ctx)?;
        let depth_test = get_ctx_depth_test(&mut ctx);
        let depth_test_comparison = DepthComparison::Less;
        let depth_write = get_ctx_depth_write(&mut ctx)?;
        let face_culling_state = get_ctx_face_culling_state(&mut ctx);
        let face_culling_order = get_ctx_face_culling_order(&mut ctx)?;
        let face_culling_mode = get_ctx_face_culling_mode(&mut ctx)?;
        let scissor_state = get_ctx_scissor_state(&mut ctx)?;
        let scissor_region = get_ctx_scissor_region(&mut ctx)?;

        let current_texture_unit = 0;
        let bound_textures = vec![(glow::TEXTURE0, None); 48]; // 48 is the platform minimal requirement
        let texture_swimming_pool = Vec::new();
        let bound_uniform_buffers = vec![None; 36]; // 36 is the platform minimal requirement
        let bound_array_buffer = None;
        let bound_element_array_buffer = None;
        let bound_draw_framebuffer = None;
        let bound_read_framebuffer = None;
        let readback_framebuffer = None;
        let bound_vertex_array = None;
        let current_program = None;

        Ok(GlowState {
            _phantom: PhantomData,
            ctx,
            binding_stack,
            viewport,
            clear_color,
            blending_state,
            blending_equations,
            blending_funcs,
            depth_test,
            depth_test_comparison,
            depth_write,
            face_culling_state,
            face_culling_order,
            face_culling_mode,
            scissor_state,
            scissor_region,
            current_texture_unit,
            bound_textures,
            texture_swimming_pool,
            bound_uniform_buffers,
            bound_array_buffer,
            bound_element_array_buffer,
            bound_draw_framebuffer,
            bound_read_framebuffer,
            readback_framebuffer,
            bound_vertex_array,
            current_program,
        })
    }

    pub(crate) fn binding_stack_mut(&mut self) -> &mut BindingStack {
        &mut self.binding_stack
    }

    pub(crate) fn create_buffer(&mut self) -> Result<glow::Buffer, String> {
        unsafe { self.ctx.create_buffer() }
    }

    pub(crate) fn bind_buffer_base(&mut self, handle: glow::Buffer, binding: u32) {
        unsafe {
            match self.bound_uniform_buffers.get(binding as usize) {
                Some(&handle_) if Some(handle) != handle_ => {
                    self.ctx
                        .bind_buffer_base(glow::UNIFORM_BUFFER, binding, Some(handle));
                    self.bound_uniform_buffers[binding as usize] = Some(handle.clone());
                }

                None => {
                    self.ctx
                        .bind_buffer_base(glow::UNIFORM_BUFFER, binding, Some(handle));

                    // not enough registered buffer bindings; let’s grow a bit more
                    self.bound_uniform_buffers
                        .resize(binding as usize + 1, None);
                    self.bound_uniform_buffers[binding as usize] = Some(handle.clone());
                }

                _ => (), // cached
            }
        }
    }

    pub(crate) fn bind_array_buffer(&mut self, buffer: Option<glow::Buffer>, bind: Bind) {
        unsafe {
            if bind == Bind::Forced || self.bound_array_buffer != buffer {
                self.ctx.bind_buffer(glow::ARRAY_BUFFER, buffer);
                self.bound_array_buffer = buffer;
            }
        }
    }

    pub(crate) fn bind_element_array_buffer(&mut self, buffer: Option<glow::Buffer>, bind: Bind) {
        unsafe {
            if bind == Bind::Forced || self.bound_element_array_buffer != buffer {
                self.ctx.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, buffer);
                self.bound_element_array_buffer = buffer;
            }
        }
    }

    pub(crate) fn unbind_buffer(&mut self, buffer: &glow::Buffer) {
        if self.bound_array_buffer.as_ref() == Some(buffer) {
            self.bind_array_buffer(None, Bind::Cached);
        } else if self.bound_element_array_buffer.as_ref() == Some(buffer) {
            self.bind_element_array_buffer(None, Bind::Cached);
        } else if let Some(handle_) = self
            .bound_uniform_buffers
            .iter_mut()
            .find(|h| h.as_ref() == Some(buffer))
        {
            *handle_ = None;
        }
    }

    pub(crate) fn create_vertex_array(&mut self) -> Result<glow::VertexArray, String> {
        unsafe { self.ctx.create_vertex_array() }
    }

    pub(crate) fn bind_vertex_array(&mut self, vao: Option<&glow::VertexArray>, bind: Bind) {
        unsafe {
            if bind == Bind::Forced || self.bound_vertex_array.as_ref() != vao {
                self.ctx.bind_vertex_array(vao.cloned());
                self.bound_vertex_array = vao.cloned();
            }
        }
    }

    pub(crate) fn create_texture(&mut self) -> Result<glow::Texture, String> {
        unsafe {
            if let Some(tex) = self.texture_swimming_pool.pop().flatten() {
                Ok(tex)
            } else {
                self.ctx.create_texture()
            }
        }
    }

    /// Reserve at least a given number of textures.
    pub(crate) fn reserve_textures(&mut self, nb: usize) {
        unsafe {
            let available = self.texture_swimming_pool.len();
            let needed = nb.max(available) - available;

            if needed > 0 {
                // resize the internal buffer to hold all the new textures and create a slice starting from
                // the previous end to the new end
                self.texture_swimming_pool.resize(available + needed, None);

                for _ in 0..needed {
                    match self.ctx.create_texture() {
                        Ok(texture) => self.texture_swimming_pool.push(Some(texture)),
                        Err(_) => break,
                    }
                }
            }
        }
    }

    pub(crate) fn set_texture_unit(&mut self, unit: u32) {
        unsafe {
            if self.current_texture_unit != unit {
                self.ctx.active_texture(glow::TEXTURE0 + unit);
                self.current_texture_unit = unit;
            }
        }
    }

    pub(crate) fn bind_texture(&mut self, target: u32, handle: Option<glow::Texture>) {
        unsafe {
            let unit = self.current_texture_unit as usize;

            match self.bound_textures.get(unit) {
                Some((t, ref h)) if target != *t || handle != *h => {
                    self.ctx.bind_texture(target, handle);
                    self.bound_textures[unit] = (target, handle);
                }

                None => {
                    self.ctx.bind_texture(target, handle);

                    // not enough available texture units; let’s grow a bit more
                    self.bound_textures
                        .resize(unit + 1, (glow::TEXTURE_2D, None));
                    self.bound_textures[unit] = (target, handle);
                }

                _ => (), // cached
            }
        }
    }

    pub(crate) fn create_framebuffer(&mut self) -> Result<glow::Framebuffer, String> {
        unsafe { self.ctx.create_framebuffer() }
    }

    pub(crate) fn create_or_get_readback_framebuffer(&mut self) -> Option<glow::Framebuffer> {
        self.readback_framebuffer.clone().or_else(|| {
            // create the readback framebuffer if not already created
            self.readback_framebuffer = self.create_framebuffer().ok();
            self.readback_framebuffer.clone()
        })
    }

    pub(crate) fn bind_draw_framebuffer(&mut self, handle: Option<glow::Framebuffer>) {
        unsafe {
            if self.bound_draw_framebuffer != handle {
                self.ctx.bind_framebuffer(glow::FRAMEBUFFER, handle);
                self.bound_draw_framebuffer = handle;
            }
        }
    }

    pub(crate) fn bind_read_framebuffer(&mut self, handle: Option<glow::Framebuffer>) {
        unsafe {
            if self.bound_read_framebuffer != handle {
                self.ctx.bind_framebuffer(glow::READ_FRAMEBUFFER, handle);
                self.bound_read_framebuffer = handle;
            }
        }
    }

    pub(crate) fn use_program(&mut self, handle: Option<glow::Program>) {
        unsafe {
            if self.current_program != handle {
                self.ctx.use_program(handle);
                self.current_program = handle;
            }
        }
    }

    pub(crate) fn set_viewport(&mut self, viewport: [i32; 4]) {
        unsafe {
            if self.viewport != viewport {
                self.ctx
                    .viewport(viewport[0], viewport[1], viewport[2], viewport[3]);
                self.viewport = viewport;
            }
        }
    }

    pub(crate) fn set_clear_color(&mut self, clear_color: [f32; 4]) {
        unsafe {
            if self.clear_color != clear_color {
                self.ctx.clear_color(
                    clear_color[0],
                    clear_color[1],
                    clear_color[2],
                    clear_color[3],
                );
                self.clear_color = clear_color;
            }
        }
    }

    pub(crate) fn set_blending_state(&mut self, state: BlendingState) {
        unsafe {
            if self.blending_state != state {
                match state {
                    BlendingState::On => self.ctx.enable(glow::BLEND),
                    BlendingState::Off => self.ctx.disable(glow::BLEND),
                }

                self.blending_state = state;
            }
        }
    }

    pub(crate) fn set_blending_equation(&mut self, equation: Equation) {
        unsafe {
            let equations = BlendingEquations {
                rgb: equation,
                alpha: equation,
            };

            if self.blending_equations != equations {
                self.ctx.blend_equation(blending_equation_to_glow(equation));
                self.blending_equations = equations;
            }
        }
    }

    pub(crate) fn set_blending_equation_separate(
        &mut self,
        equation_rgb: Equation,
        equation_alpha: Equation,
    ) {
        unsafe {
            let equations = BlendingEquations {
                rgb: equation_rgb,
                alpha: equation_alpha,
            };

            if self.blending_equations != equations {
                self.ctx.blend_equation_separate(
                    blending_equation_to_glow(equation_rgb),
                    blending_equation_to_glow(equation_alpha),
                );

                self.blending_equations = equations;
            }
        }
    }

    pub(crate) fn set_blending_func(&mut self, src: Factor, dst: Factor) {
        unsafe {
            let funcs = BlendingFactors {
                src_rgb: src,
                dst_rgb: dst,
                src_alpha: src,
                dst_alpha: dst,
            };

            if self.blending_funcs != funcs {
                self.ctx
                    .blend_func(blending_factor_to_glow(src), blending_factor_to_glow(dst));

                self.blending_funcs = funcs;
            }
        }
    }

    pub(crate) fn set_blending_func_separate(
        &mut self,
        src_rgb: Factor,
        dst_rgb: Factor,
        src_alpha: Factor,
        dst_alpha: Factor,
    ) {
        unsafe {
            let funcs = BlendingFactors {
                src_rgb,
                dst_rgb,
                src_alpha,
                dst_alpha,
            };
            if self.blending_funcs != funcs {
                self.ctx.blend_func_separate(
                    blending_factor_to_glow(src_rgb),
                    blending_factor_to_glow(dst_rgb),
                    blending_factor_to_glow(src_alpha),
                    blending_factor_to_glow(dst_alpha),
                );

                self.blending_funcs = funcs;
            }
        }
    }

    pub(crate) fn set_depth_test(&mut self, depth_test: DepthTest) {
        unsafe {
            if self.depth_test != depth_test {
                match depth_test {
                    DepthTest::On => self.ctx.enable(glow::DEPTH_TEST),
                    DepthTest::Off => self.ctx.disable(glow::DEPTH_TEST),
                }

                self.depth_test = depth_test;
            }
        }
    }

    pub(crate) fn set_depth_test_comparison(&mut self, depth_test_comparison: DepthComparison) {
        unsafe {
            if self.depth_test_comparison != depth_test_comparison {
                self.ctx
                    .depth_func(depth_comparison_to_glow(depth_test_comparison));

                self.depth_test_comparison = depth_test_comparison;
            }
        }
    }

    pub(crate) fn set_depth_write(&mut self, depth_write: DepthWrite) {
        unsafe {
            if self.depth_write != depth_write {
                let enabled = match depth_write {
                    DepthWrite::On => true,
                    DepthWrite::Off => false,
                };

                self.ctx.depth_mask(enabled);

                self.depth_write = depth_write;
            }
        }
    }

    pub(crate) fn set_face_culling_state(&mut self, state: FaceCullingState) {
        unsafe {
            if self.face_culling_state != state {
                match state {
                    FaceCullingState::On => self.ctx.enable(glow::CULL_FACE),
                    FaceCullingState::Off => self.ctx.disable(glow::CULL_FACE),
                }

                self.face_culling_state = state;
            }
        }
    }

    pub(crate) fn set_face_culling_order(&mut self, order: FaceCullingOrder) {
        unsafe {
            if self.face_culling_order != order {
                match order {
                    FaceCullingOrder::CW => self.ctx.front_face(glow::CW),
                    FaceCullingOrder::CCW => self.ctx.front_face(glow::CCW),
                }

                self.face_culling_order = order;
            }
        }
    }

    pub(crate) fn set_face_culling_mode(&mut self, mode: FaceCullingMode) {
        unsafe {
            if self.face_culling_mode != mode {
                match mode {
                    FaceCullingMode::Front => self.ctx.cull_face(glow::FRONT),
                    FaceCullingMode::Back => self.ctx.cull_face(glow::BACK),
                    FaceCullingMode::Both => self.ctx.cull_face(glow::FRONT_AND_BACK),
                }

                self.face_culling_mode = mode;
            }
        }
    }

    pub(crate) fn set_scissor_state(&mut self, state: ScissorState) {
        unsafe {
            if self.scissor_state != state {
                match state {
                    ScissorState::On => self.ctx.enable(glow::SCISSOR_TEST),
                    ScissorState::Off => self.ctx.disable(glow::SCISSOR_TEST),
                }

                self.scissor_state = state;
            }
        }
    }

    pub(crate) fn set_scissor_region(&mut self, region: &ScissorRegion) {
        unsafe {
            if self.scissor_region != *region {
                let ScissorRegion {
                    x,
                    y,
                    width,
                    height,
                } = *region;

                self.ctx
                    .scissor(x as i32, y as i32, width as i32, height as i32);
                self.scissor_region = *region;
            }
        }
    }
}

impl Drop for GlowState {
    fn drop(&mut self) {
        unsafe {
            // drop the readback framebuffer if it was allocated
            self.readback_framebuffer
                .map(|x| self.ctx.delete_framebuffer(x));
        }
    }
}

/// An error that might happen when the context is queried.
#[non_exhaustive]
#[derive(Debug)]
pub enum StateQueryError {
    /// The [`GlowState`] object is unavailable.
    ///
    /// That might occur if the current thread doesn’t support allocating a new graphics state. It
    /// might happen if you try to have more than one state on the same thread, for instance.
    ///
    /// [`GlowState`]: crate::glow2::state::GlowState
    UnavailableGlowState,
    /// Unknown array buffer initial state.
    UnknownArrayBufferInitialState,
    /// Unknown viewport initial state.
    UnknownViewportInitialState,
    /// Unknown clear color initial state.
    UnknownClearColorInitialState,
    /// Unknown depth write mask initial state.
    UnknownDepthWriteMaskState,
    /// Corrupted blending equation.
    UnknownBlendingEquation(u32),
    /// RGB blending equation couldn’t be retrieved when initializing the Glow state.
    CannotRetrieveBlendingEquationRGB,
    /// Alpha blending equation couldn’t be retrieved when initializing the Glow state.
    CannotRetrieveBlendingEquationAlpha,
    /// Source RGB factor couldn’t be retrieved when initializing the Glow state.
    CannotRetrieveBlendingSrcFactorRGB,
    /// Source alpha factor couldn’t be retrieved when initializing the Glow state.
    CannotRetrieveBlendingSrcFactorAlpha,
    /// Destination RGB factor couldn’t be retrieved when initializing the Glow state.
    CannotRetrieveBlendingDstFactorRGB,
    /// Destination alpha factor couldn’t be retrieved when initializing the Glow state.
    CannotRetrieveBlendingDstFactorAlpha,
    /// Required WebGL extensions cannot be enabled
    CannotRetrieveRequiredGlowExtensions(Vec<String>),
    /// Corrupted blending source factor (RGB).
    UnknownBlendingSrcFactorRGB(u32),
    /// Corrupted blending source factor (alpha).
    UnknownBlendingSrcFactorAlpha(u32),
    /// Corrupted blending destination factor (RGB).
    UnknownBlendingDstFactorRGB(u32),
    /// Corrupted blending destination factor (alpha).
    UnknownBlendingDstFactorAlpha(u32),
    /// Corrupted face culling order.
    UnknownFaceCullingOrder,
    /// Corrupted face culling mode.
    UnknownFaceCullingMode,
    /// Unknown scissor region initial state.
    UnknownScissorRegionInitialState,
}

impl fmt::Display for StateQueryError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            StateQueryError::UnavailableGlowState => write!(f, "unavailable graphics state"),

            StateQueryError::UnknownArrayBufferInitialState => {
                write!(f, "unknown array buffer initial state")
            }

            StateQueryError::UnknownViewportInitialState => {
                write!(f, "unknown viewport initial state")
            }

            StateQueryError::UnknownClearColorInitialState => {
                write!(f, "unknown clear color initial state")
            }

            StateQueryError::UnknownDepthWriteMaskState => {
                f.write_str("unkonwn depth write mask state")
            }

            StateQueryError::UnknownBlendingEquation(ref e) => {
                write!(f, "unknown blending equation: {}", e)
            }

            StateQueryError::CannotRetrieveBlendingEquationRGB => {
                f.write_str("cannot retrieve blending equation (RGB)")
            }

            StateQueryError::CannotRetrieveBlendingEquationAlpha => {
                f.write_str("cannot retrieve blending equation (alpha)")
            }

            StateQueryError::CannotRetrieveBlendingSrcFactorRGB => {
                f.write_str("cannot retrieve blending source factor (RGB)")
            }

            StateQueryError::CannotRetrieveBlendingSrcFactorAlpha => {
                f.write_str("cannot retrieve blending source factor (alpha)")
            }

            StateQueryError::CannotRetrieveBlendingDstFactorRGB => {
                f.write_str("cannot retrieve blending destination factor (RGB)")
            }

            StateQueryError::CannotRetrieveBlendingDstFactorAlpha => {
                f.write_str("cannot retrieve blending destination factor (alpha)")
            }

            StateQueryError::CannotRetrieveRequiredGlowExtensions(ref extensions) => write!(
                f,
                "missing Glow extensions: [{}]",
                extensions.join(", ").as_str()
            ),

            StateQueryError::UnknownBlendingSrcFactorRGB(ref k) => {
                write!(f, "unknown blending source factor (RGB): {}", k)
            }

            StateQueryError::UnknownBlendingSrcFactorAlpha(ref k) => {
                write!(f, "unknown blending source factor (alpha): {}", k)
            }

            StateQueryError::UnknownBlendingDstFactorRGB(ref k) => {
                write!(f, "unknown blending destination factor (RGB): {}", k)
            }

            StateQueryError::UnknownBlendingDstFactorAlpha(ref k) => {
                write!(f, "unknown blending destination factor (alpha): {}", k)
            }

            StateQueryError::UnknownFaceCullingOrder => f.write_str("unknown face culling order"),

            StateQueryError::UnknownFaceCullingMode => f.write_str("unknown face culling mode"),

            StateQueryError::UnknownScissorRegionInitialState => {
                write!(f, "unknown scissor region initial state")
            }
        }
    }
}

impl std::error::Error for StateQueryError {}

fn get_ctx_viewport(ctx: &mut glow::Context) -> Result<[i32; 4], StateQueryError> {
    let mut viewport = [0; 4];

    unsafe { ctx.get_parameter_i32_slice(glow::VIEWPORT, &mut viewport) };

    Ok(viewport)
}

fn get_ctx_clear_color(ctx: &mut glow::Context) -> Result<[f32; 4], StateQueryError> {
    let mut color = [0.0; 4];

    unsafe { ctx.get_parameter_f32_slice(glow::COLOR_CLEAR_VALUE, &mut color) };

    Ok(color)
}

fn get_ctx_blending_state(ctx: &mut glow::Context) -> BlendingState {
    unsafe {
        if ctx.is_enabled(glow::BLEND) {
            BlendingState::On
        } else {
            BlendingState::Off
        }
    }
}

fn get_ctx_blending_equations(
    ctx: &mut glow::Context,
) -> Result<BlendingEquations, StateQueryError> {
    unsafe {
        let rgb =
            map_enum_to_blending_equation(ctx.get_parameter_i32(glow::BLEND_EQUATION_RGB) as u32)?;

        let alpha = map_enum_to_blending_equation(
            ctx.get_parameter_i32(glow::BLEND_EQUATION_ALPHA) as u32,
        )?;

        Ok(BlendingEquations { rgb, alpha })
    }
}

#[inline]
fn map_enum_to_blending_equation(data: u32) -> Result<Equation, StateQueryError> {
    match data {
        glow::FUNC_ADD => Ok(Equation::Additive),
        glow::FUNC_SUBTRACT => Ok(Equation::Subtract),
        glow::FUNC_REVERSE_SUBTRACT => Ok(Equation::ReverseSubtract),
        glow::MIN => Ok(Equation::Min),
        glow::MAX => Ok(Equation::Max),
        _ => Err(StateQueryError::UnknownBlendingEquation(data)),
    }
}

fn get_ctx_blending_factors(ctx: &mut glow::Context) -> Result<BlendingFactors, StateQueryError> {
    unsafe {
        let src_rgb = ctx.get_parameter_i32(glow::BLEND_SRC_RGB) as u32;
        let src_rgb = from_gl_blending_factor(src_rgb)
            .map_err(StateQueryError::UnknownBlendingSrcFactorRGB)?;

        let src_alpha = ctx.get_parameter_i32(glow::BLEND_SRC_ALPHA) as u32;
        let src_alpha = from_gl_blending_factor(src_alpha)
            .map_err(StateQueryError::UnknownBlendingSrcFactorAlpha)?;

        let dst_rgb = ctx.get_parameter_i32(glow::BLEND_DST_RGB) as u32;
        let dst_rgb = from_gl_blending_factor(dst_rgb)
            .map_err(StateQueryError::UnknownBlendingDstFactorRGB)?;

        let dst_alpha = ctx.get_parameter_i32(glow::BLEND_DST_ALPHA) as u32;
        let dst_alpha = from_gl_blending_factor(dst_alpha)
            .map_err(StateQueryError::UnknownBlendingDstFactorAlpha)?;

        Ok(BlendingFactors {
            src_rgb,
            dst_rgb,
            src_alpha,
            dst_alpha,
        })
    }
}

#[inline]
fn from_gl_blending_factor(factor: u32) -> Result<Factor, u32> {
    match factor {
        glow::ONE => Ok(Factor::One),
        glow::ZERO => Ok(Factor::Zero),
        glow::SRC_COLOR => Ok(Factor::SrcColor),
        glow::ONE_MINUS_SRC_COLOR => Ok(Factor::SrcColorComplement),
        glow::DST_COLOR => Ok(Factor::DestColor),
        glow::ONE_MINUS_DST_COLOR => Ok(Factor::DestColorComplement),
        glow::SRC_ALPHA => Ok(Factor::SrcAlpha),
        glow::ONE_MINUS_SRC_ALPHA => Ok(Factor::SrcAlphaComplement),
        glow::DST_ALPHA => Ok(Factor::DstAlpha),
        glow::ONE_MINUS_DST_ALPHA => Ok(Factor::DstAlphaComplement),
        glow::SRC_ALPHA_SATURATE => Ok(Factor::SrcAlphaSaturate),
        _ => Err(factor),
    }
}

fn get_ctx_depth_test(ctx: &mut glow::Context) -> DepthTest {
    unsafe {
        let enabled = ctx.is_enabled(glow::DEPTH_TEST);

        if enabled {
            DepthTest::On
        } else {
            DepthTest::Off
        }
    }
}

fn get_ctx_depth_write(ctx: &mut glow::Context) -> Result<DepthWrite, StateQueryError> {
    unsafe {
        let enabled = ctx.get_parameter_i32(glow::DEPTH_WRITEMASK) != 0;

        if enabled {
            Ok(DepthWrite::On)
        } else {
            Ok(DepthWrite::Off)
        }
    }
}

fn get_ctx_face_culling_state(ctx: &mut glow::Context) -> FaceCullingState {
    unsafe {
        let enabled = ctx.is_enabled(glow::CULL_FACE);

        if enabled {
            FaceCullingState::On
        } else {
            FaceCullingState::Off
        }
    }
}

fn get_ctx_face_culling_order(
    ctx: &mut glow::Context,
) -> Result<FaceCullingOrder, StateQueryError> {
    unsafe {
        let order = ctx.get_parameter_i32(glow::FRONT_FACE) as u32;

        match order {
            glow::CCW => Ok(FaceCullingOrder::CCW),
            glow::CW => Ok(FaceCullingOrder::CW),
            _ => Err(StateQueryError::UnknownFaceCullingOrder),
        }
    }
}

fn get_ctx_face_culling_mode(ctx: &mut glow::Context) -> Result<FaceCullingMode, StateQueryError> {
    unsafe {
        let mode = ctx.get_parameter_i32(glow::CULL_FACE_MODE) as u32;

        match mode {
            glow::FRONT => Ok(FaceCullingMode::Front),
            glow::BACK => Ok(FaceCullingMode::Back),
            glow::FRONT_AND_BACK => Ok(FaceCullingMode::Both),
            _ => Err(StateQueryError::UnknownFaceCullingMode),
        }
    }
}

fn get_ctx_scissor_state(ctx: &mut glow::Context) -> Result<ScissorState, StateQueryError> {
    unsafe {
        let state = if ctx.is_enabled(glow::SCISSOR_TEST) {
            ScissorState::On
        } else {
            ScissorState::Off
        };

        Ok(state)
    }
}

fn get_ctx_scissor_region(ctx: &mut glow::Context) -> Result<ScissorRegion, StateQueryError> {
    unsafe {
        let mut region = [0; 4];
        ctx.get_parameter_i32_slice(glow::SCISSOR_BOX, &mut region);

        Ok(ScissorRegion {
            x: region[0] as u32,
            y: region[1] as u32,
            width: region[2] as u32,
            height: region[3] as u32,
        })
    }
}

// I think that Glow handles this for us, though I don't know exactly what will happen if we don't
// have the exensions that we need.
//
// fn load_glow2_extensions(ctx: &mut glow::Context) -> Result<(), StateQueryError> {
//     let required_extensions = [
//         "OES_texture_float_linear",
//         "EXT_color_buffer_float",
//         "EXT_float_blend",
//     ];

//     let available_extensions: Vec<&str> = required_extensions
//         .iter()
//         .map(|ext| (*ext, ctx.get_extension(ext)))
//         .flat_map(|(ext, result)| result.ok().flatten().map(|_| ext))
//         .collect();

//     if available_extensions.len() < required_extensions.len() {
//         let missing_extensions: Vec<String> = required_extensions
//             .iter()
//             .filter(|e| !available_extensions.contains(e))
//             .map(|e| e.to_string())
//             .collect();

//         return Err(StateQueryError::CannotRetrieveRequiredGlowExtensions(
//             missing_extensions,
//         ));
//     }

//     Ok(())
// }

/// Should the binding be cached or forced to the provided value?
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum Bind {
    Forced,
    Cached,
}

/// Whether or not enable blending.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum BlendingState {
    /// Enable blending.
    On,
    /// Disable blending.
    Off,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct BlendingFactors {
    src_rgb: Factor,
    dst_rgb: Factor,
    src_alpha: Factor,
    dst_alpha: Factor,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct BlendingEquations {
    rgb: Equation,
    alpha: Equation,
}

/// Whether or not depth test should be enabled.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum DepthTest {
    /// The depth test is enabled.
    On,
    /// The depth test is disabled.
    Off,
}

/// Should face culling be enabled?
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum FaceCullingState {
    /// Enable face culling.
    On,
    /// Disable face culling.
    Off,
}

#[inline]
fn depth_comparison_to_glow(dc: DepthComparison) -> u32 {
    match dc {
        DepthComparison::Never => glow::NEVER,
        DepthComparison::Always => glow::ALWAYS,
        DepthComparison::Equal => glow::EQUAL,
        DepthComparison::NotEqual => glow::NOTEQUAL,
        DepthComparison::Less => glow::LESS,
        DepthComparison::LessOrEqual => glow::LEQUAL,
        DepthComparison::Greater => glow::GREATER,
        DepthComparison::GreaterOrEqual => glow::GEQUAL,
    }
}

#[inline]
fn blending_equation_to_glow(equation: Equation) -> u32 {
    match equation {
        Equation::Additive => glow::FUNC_ADD,
        Equation::Subtract => glow::FUNC_SUBTRACT,
        Equation::ReverseSubtract => glow::FUNC_REVERSE_SUBTRACT,
        Equation::Min => glow::MIN,
        Equation::Max => glow::MAX,
    }
}

#[inline]
fn blending_factor_to_glow(factor: Factor) -> u32 {
    match factor {
        Factor::One => glow::ONE,
        Factor::Zero => glow::ZERO,
        Factor::SrcColor => glow::SRC_COLOR,
        Factor::SrcColorComplement => glow::ONE_MINUS_SRC_COLOR,
        Factor::DestColor => glow::DST_COLOR,
        Factor::DestColorComplement => glow::ONE_MINUS_DST_COLOR,
        Factor::SrcAlpha => glow::SRC_ALPHA,
        Factor::SrcAlphaComplement => glow::ONE_MINUS_SRC_ALPHA,
        Factor::DstAlpha => glow::DST_ALPHA,
        Factor::DstAlphaComplement => glow::ONE_MINUS_DST_ALPHA,
        Factor::SrcAlphaSaturate => glow::SRC_ALPHA_SATURATE,
    }
}

/// Scissor state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ScissorState {
    /// Enabled.
    On,
    /// Disabled
    Off,
}
