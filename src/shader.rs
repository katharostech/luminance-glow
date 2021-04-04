//! Shader support for Glow.

use luminance::backend::shader::{Shader, Uniformable};
use luminance::pipeline::{BufferBinding, TextureBinding};
use luminance::pixel::{SamplerType, Type as PixelType};
use luminance::shader::{
    ProgramError, StageError, StageType, TessellationStages, Uniform, UniformType, UniformWarning,
    VertexAttribWarning,
};
use luminance::texture::{Dim, Dimensionable};
use luminance::vertex::Semantics;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::state::GlowState;
use crate::Glow;
use glow::HasContext;

#[derive(Debug)]
pub struct Stage {
    handle: glow::Shader,
    ty: StageType,
    state: Rc<RefCell<GlowState>>,
}

impl Drop for Stage {
    fn drop(&mut self) {
        unsafe {
            self.state.borrow().ctx.delete_shader(self.handle);
        }
    }
}

impl Stage {
    fn new(glow_backend: &mut Glow, ty: StageType, src: &str) -> Result<Self, StageError> {
        unsafe {
            let state = glow_backend.state.borrow();

            let shader_ty = glow_shader_type(ty).ok_or_else(|| {
                StageError::CompilationFailed(ty, "unsupported shader type".to_owned())
            })?;

            let handle = state.ctx.create_shader(shader_ty).map_err(|e| {
                StageError::CompilationFailed(ty, format!("unable to create shader stage: {}", e))
            })?;

            state
                .ctx
                .shader_source(handle, &patch_shader_src(src, glow_backend.is_webgl1));
            state.ctx.compile_shader(handle);

            let compiled = state.ctx.get_shader_compile_status(handle);

            if compiled {
                Ok(Stage {
                    handle,
                    ty,
                    state: glow_backend.state.clone(),
                })
            } else {
                let log = state.ctx.get_shader_info_log(handle);

                state.ctx.delete_shader(handle);

                Err(StageError::compilation_failed(ty, log))
            }
        }
    }

    fn handle(&self) -> &glow::Shader {
        &self.handle
    }
}

/// A type used to map [`i32`] (uniform locations) to [`WebGlUniformLocation`].
///
/// It is typically shared with internal mutation (Rc + RefCell) so that it can add the location
/// mappings in the associated [`Program`].
type LocationMap = HashMap<i32, glow::UniformLocation>;

#[derive(Debug)]
pub struct Program {
    pub(crate) handle: glow::Program,
    location_map: Rc<RefCell<LocationMap>>,
    state: Rc<RefCell<GlowState>>,
}

impl Drop for Program {
    fn drop(&mut self) {
        unsafe {
            self.state.borrow().ctx.delete_program(self.handle);
        }
    }
}

impl Program {
    fn new(
        glow2: &mut Glow,
        vertex: &Stage,
        tess: Option<TessellationStages<Stage>>,
        geometry: Option<&Stage>,
        fragment: &Stage,
    ) -> Result<Self, ProgramError> {
        unsafe {
            let state = glow2.state.borrow();

            let handle = state.ctx.create_program().map_err(|e| {
                ProgramError::CreationFailed(format!(
                    "unable to allocate GPU shader program: {}",
                    e
                ))
            })?;

            if let Some(TessellationStages {
                control,
                evaluation,
            }) = tess
            {
                state.ctx.attach_shader(handle, *control.handle());
                state.ctx.attach_shader(handle, *evaluation.handle());
            }

            state.ctx.attach_shader(handle, *vertex.handle());

            if let Some(geometry) = geometry {
                state.ctx.attach_shader(handle, *geometry.handle());
            }

            state.ctx.attach_shader(handle, *fragment.handle());

            let location_map = Rc::new(RefCell::new(HashMap::new()));
            let state = glow2.state.clone();
            let program = Program {
                handle,
                location_map,
                state,
            };

            program.link().map(move |_| program)
        }
    }

    fn link(&self) -> Result<(), ProgramError> {
        unsafe {
            let handle = self.handle;
            let state = self.state.borrow();

            state.ctx.link_program(handle);

            let linked = state.ctx.get_program_link_status(handle);

            if linked {
                Ok(())
            } else {
                let log = state.ctx.get_program_info_log(handle);
                Err(ProgramError::link_failed(log))
            }
        }
    }

    fn handle(&self) -> &glow::Program {
        &self.handle
    }
}

pub struct UniformBuilder {
    handle: glow::Program,
    location_map: Rc<RefCell<LocationMap>>,
    state: Rc<RefCell<GlowState>>,
}

impl UniformBuilder {
    fn new(program: &Program) -> Self {
        UniformBuilder {
            handle: program.handle.clone(),
            location_map: program.location_map.clone(),
            state: program.state.clone(),
        }
    }

    fn ask_uniform<T>(&mut self, name: &str) -> Result<Uniform<T>, UniformWarning>
    where
        T: Uniformable<Glow>,
    {
        unsafe {
            let location = self
                .state
                .borrow()
                .ctx
                .get_uniform_location(self.handle, name);

            match location {
                Some(location) => {
                    // if we correctly map the uniform, we generate a new ID by using the length of the current
                    // location map — we never remove items, so it will always grow — and insert the indirection
                    let mut location_map = self.location_map.borrow_mut();
                    let idx = location_map.len() as i32;
                    location_map.insert(idx, location);

                    Ok(Uniform::new(idx))
                }

                None => Err(UniformWarning::inactive(name)),
            }
        }
    }

    fn ask_uniform_block<T>(&self, name: &str) -> Result<Uniform<T>, UniformWarning>
    where
        T: Uniformable<Glow>,
    {
        unsafe {
            let location = self
                .state
                .borrow()
                .ctx
                .get_uniform_block_index(self.handle, name);

            Ok(location
                .map(|x| Uniform::new(x as i32))
                .ok_or(UniformWarning::inactive(name))?)
        }
    }
}

unsafe impl Shader for Glow {
    type StageRepr = Stage;

    type ProgramRepr = Program;

    type UniformBuilderRepr = UniformBuilder;

    unsafe fn new_stage(
        &mut self,
        ty: StageType,
        src: &str,
    ) -> Result<Self::StageRepr, StageError> {
        Stage::new(self, ty, src)
    }

    unsafe fn new_program(
        &mut self,
        vertex: &Self::StageRepr,
        tess: Option<TessellationStages<Self::StageRepr>>,
        geometry: Option<&Self::StageRepr>,
        fragment: &Self::StageRepr,
    ) -> Result<Self::ProgramRepr, ProgramError> {
        Program::new(self, vertex, tess, geometry, fragment)
    }

    unsafe fn apply_semantics<Sem>(
        program: &mut Self::ProgramRepr,
    ) -> Result<Vec<VertexAttribWarning>, ProgramError>
    where
        Sem: Semantics,
    {
        let warnings = {
            let state = program.state.borrow();
            bind_vertex_attribs_locations::<Sem>(&state, program)
        };

        // we need to link again to make the location mappings a thing
        program.link()?;
        Ok(warnings)
    }

    unsafe fn new_uniform_builder(
        program: &mut Self::ProgramRepr,
    ) -> Result<Self::UniformBuilderRepr, ProgramError> {
        Ok(UniformBuilder::new(&program))
    }

    unsafe fn ask_uniform<T>(
        uniform_builder: &mut Self::UniformBuilderRepr,
        name: &str,
    ) -> Result<Uniform<T>, UniformWarning>
    where
        T: Uniformable<Self>,
    {
        let ty = T::ty();
        let uniform = match ty {
            UniformType::BufferBinding => uniform_builder.ask_uniform_block(name)?,
            _ => uniform_builder.ask_uniform(name)?,
        };

        let state = uniform_builder.state.borrow();
        uniform_type_match(&state, &uniform_builder.handle, name, ty)?;

        Ok(uniform)
    }

    unsafe fn unbound<T>(_: &mut Self::UniformBuilderRepr) -> Uniform<T>
    where
        T: Uniformable<Self>,
    {
        Uniform::new(-1)
    }
}

fn glow_shader_type(ty: StageType) -> Option<u32> {
    match ty {
        StageType::VertexShader => Some(glow::VERTEX_SHADER),
        StageType::FragmentShader => Some(glow::FRAGMENT_SHADER),
        _ => None,
    }
}

const WEBGL1_GLSL_PRAGMA: &str = "#version 100\n\
                           precision highp float;\n\
                           precision highp int;\n";
const GLSL_PRAGMA: &str = "#version 300 es\n\
                           precision highp float;\n\
                           precision highp int;\n";

fn patch_shader_src(src: &str, is_webgl1: bool) -> String {
    let mut pragma;
    if is_webgl1 {
        pragma = String::from(WEBGL1_GLSL_PRAGMA);
    } else {
        pragma = String::from(GLSL_PRAGMA);
    }
    pragma.push_str(src);
    pragma
}

fn uniform_type_match(
    _state: &GlowState,
    _program: &glow::Program,
    _name: &str,
    _ty: UniformType,
) -> Result<(), UniformWarning> {
    // unsafe {
    // // get the index of the uniform; it’s represented as an array of a single element, since our
    // // input has only one element
    // let index = state
    //     .ctx
    //     .get_uniform_location(*program, name)
    //     .ok_or_else(|| {
    //         UniformWarning::TypeMismatch("cannot retrieve uniform index".to_owned(), ty)
    //     })?;

    // // get its size and type
    // let info = state
    //     .ctx
    //     .get_active_uniform(*program, index)
    //     .ok_or_else(|| {
    //         UniformWarning::TypeMismatch("cannot retrieve active uniform".to_owned(), ty)
    //     })?;

    // check_types_match(name, ty, info.utype)
    // FIXME: I don't know how we can actually get the uniform info from the name yet
    Ok(())
    // }
}

// #[allow(clippy::cognitive_complexity)]
// fn check_types_match(name: &str, ty: UniformType, glty: u32) -> Result<(), UniformWarning> {
//     // helper macro to check type mismatch for each variant
//     macro_rules! milkcheck {
//     ($ty:expr, $( ( $v:tt, $t:tt ) ),*) => {
//       match $ty {
//         $(
//           UniformType::$v => {
//             if glty == glow::$t {
//               Ok(())
//             } else {
//               Err(UniformWarning::type_mismatch(name, ty))
//             }
//           }
//         )*

//         _ => Err(UniformWarning::unsupported_type(name, ty))
//       }
//     }
//   }

//     milkcheck!(
//         ty,
//         // scalars
//         (Int, INT),
//         (UInt, UNSIGNED_INT),
//         (Float, FLOAT),
//         (Bool, BOOL),
//         // vectors
//         (IVec2, INT_VEC2),
//         (IVec3, INT_VEC3),
//         (IVec4, INT_VEC4),
//         (UIVec2, UNSIGNED_INT_VEC2),
//         (UIVec3, UNSIGNED_INT_VEC3),
//         (UIVec4, UNSIGNED_INT_VEC4),
//         (Vec2, FLOAT_VEC2),
//         (Vec3, FLOAT_VEC3),
//         (Vec4, FLOAT_VEC4),
//         (BVec2, BOOL_VEC2),
//         (BVec3, BOOL_VEC3),
//         (BVec4, BOOL_VEC4),
//         // matrices
//         (M22, FLOAT_MAT2),
//         (M33, FLOAT_MAT3),
//         (M44, FLOAT_MAT4),
//         // textures
//         (ISampler2D, INT_SAMPLER_2D),
//         (ISampler3D, INT_SAMPLER_3D),
//         (ISampler2DArray, INT_SAMPLER_2D_ARRAY),
//         (UISampler2D, UNSIGNED_INT_SAMPLER_2D),
//         (UISampler3D, UNSIGNED_INT_SAMPLER_3D),
//         (UISampler2DArray, UNSIGNED_INT_SAMPLER_2D_ARRAY),
//         (Sampler2D, SAMPLER_2D),
//         (Sampler3D, SAMPLER_3D),
//         (Sampler2DArray, SAMPLER_2D_ARRAY),
//         (ICubemap, INT_SAMPLER_CUBE),
//         (UICubemap, UNSIGNED_INT_SAMPLER_CUBE),
//         (Cubemap, SAMPLER_CUBE)
//     )
// }

fn bind_vertex_attribs_locations<Sem>(
    state: &GlowState,
    program: &Program,
) -> Vec<VertexAttribWarning>
where
    Sem: Semantics,
{
    let mut warnings = Vec::new();

    for desc in Sem::semantics_set() {
        match get_vertex_attrib_location(state, program, &desc.name) {
            Ok(_) => {
                let index = desc.index as u32;
                unsafe {
                    state
                        .ctx
                        .bind_attrib_location(*program.handle(), index, &desc.name);
                }
            }

            Err(warning) => warnings.push(warning),
        }
    }

    warnings
}

fn get_vertex_attrib_location(
    state: &GlowState,
    program: &Program,
    name: &str,
) -> Result<i32, VertexAttribWarning> {
    unsafe {
        let location = state.ctx.get_attrib_location(*program.handle(), name);

        if let Some(location) = location {
            Ok(location as i32)
        } else {
            Err(VertexAttribWarning::inactive(name))
        }
    }
}

// A helper macro used to implement Uniformable for very similar function types. Several forms
// exist: mostly, slice form (like &[_]) will have to compute a length and pass the length down the
// WebGL function along with a flatten version of a slice, while scalar versions will simply
// forward the arguments.
//
// The matrix form is ugly: the mat T ; N form is simply akin to [[T; N]; N] while the
// mat & T ; N is akin to &[[[T; N]; N]] (i.e. a slice of matrices).
//
// Also, ideally, the ty() function should live in the luminance crate, because other backends will
// have to copy the same implementation over and over.
//
// I’m so sorry.
macro_rules! impl_Uniformable {
    (&[[$t:ty; $dim:expr]], $uty:tt, $f:tt) => {
        unsafe impl<'a> Uniformable<Glow> for &'a [[$t; $dim]] {
            unsafe fn ty() -> UniformType {
                UniformType::$uty
            }

            unsafe fn update(self, program: &mut Program, uniform: &Uniform<Self>) {
                let data = flatten_slice!(self: $t, len = $dim * self.len());

                program
                    .state
                    .borrow()
                    .ctx
                    .$f(program.location_map.borrow().get(&uniform.index()), data);
            }
        }
    };

    (&[$t:ty], $uty:tt, $f:tt) => {
        unsafe impl<'a> Uniformable<Glow> for &'a [$t] {
            unsafe fn ty() -> UniformType {
                UniformType::$uty
            }

            unsafe fn update(self, program: &mut Program, uniform: &Uniform<Self>) {
                program
                    .state
                    .borrow()
                    .ctx
                    .$f(program.location_map.borrow().get(&uniform.index()), self);
            }
        }
    };

    ([$t:ty; 1], $uty:tt, $f:tt) => {
        unsafe impl Uniformable<Glow> for [$t; 1] {
            unsafe fn ty() -> UniformType {
                UniformType::$uty
            }

            unsafe fn update(self, program: &mut Program, uniform: &Uniform<Self>) {
                program
                    .state
                    .borrow()
                    .ctx
                    .$f(program.location_map.borrow().get(&uniform.index()), self[1]);
            }
        }
    };

    ([$t:ty; 2], $uty:tt, $f:tt) => {
        unsafe impl Uniformable<Glow> for [$t; 2] {
            unsafe fn ty() -> UniformType {
                UniformType::$uty
            }

            unsafe fn update(self, program: &mut Program, uniform: &Uniform<Self>) {
                program.state.borrow().ctx.$f(
                    program.location_map.borrow().get(&uniform.index()),
                    self[0],
                    self[1],
                );
            }
        }
    };

    ([$t:ty; 3], $uty:tt, $f:tt) => {
        unsafe impl Uniformable<Glow> for [$t; 3] {
            unsafe fn ty() -> UniformType {
                UniformType::$uty
            }

            unsafe fn update(self, program: &mut Program, uniform: &Uniform<Self>) {
                program.state.borrow().ctx.$f(
                    program.location_map.borrow().get(&uniform.index()),
                    self[0],
                    self[1],
                    self[2],
                );
            }
        }
    };

    ([$t:ty; 4], $uty:tt, $f:tt) => {
        unsafe impl Uniformable<Glow> for [$t; 4] {
            unsafe fn ty() -> UniformType {
                UniformType::$uty
            }

            unsafe fn update(self, program: &mut Program, uniform: &Uniform<Self>) {
                program.state.borrow().ctx.$f(
                    program.location_map.borrow().get(&uniform.index()),
                    self[0],
                    self[1],
                    self[2],
                    self[3],
                );
            }
        }
    };

    ($t:ty, $uty:tt, $f:tt) => {
        unsafe impl Uniformable<Glow> for $t {
            unsafe fn ty() -> UniformType {
                UniformType::$uty
            }

            unsafe fn update(self, program: &mut Program, uniform: &Uniform<Self>) {
                program
                    .state
                    .borrow()
                    .ctx
                    .$f(program.location_map.borrow().get(&uniform.index()), self);
            }
        }
    };

    // matrix notation
    (mat & $t:ty ; $dim:expr, $uty:tt, $f:tt) => {
        unsafe impl<'a> Uniformable<Glow> for &'a [[[$t; $dim]; $dim]] {
            unsafe fn ty() -> UniformType {
                UniformType::$uty
            }

            unsafe fn update(self, program: &mut Program, uniform: &Uniform<Self>) {
                let data = flatten_slice!(self: $t, len = $dim * $dim * self.len());

                program.state.borrow().ctx.$f(
                    program.location_map.borrow().get(&uniform.index()),
                    false,
                    data,
                );
            }
        }
    };

    (mat $t:ty ; $dim:expr, $uty:tt, $f:tt) => {
        unsafe impl Uniformable<Glow> for [[$t; $dim]; $dim] {
            unsafe fn ty() -> UniformType {
                UniformType::$uty
            }

            unsafe fn update(self, program: &mut Program, uniform: &Uniform<Self>) {
                let data = flatten_slice!(self: $t, len = $dim * $dim);

                program.state.borrow().ctx.$f(
                    program.location_map.borrow().get(&uniform.index()),
                    false,
                    data,
                );
            }
        }
    };
}

// here we go in deep mud
impl_Uniformable!(i32, Int, uniform_1_i32);
impl_Uniformable!([i32; 2], IVec2, uniform_2_i32);
impl_Uniformable!([i32; 3], IVec3, uniform_3_i32);
impl_Uniformable!([i32; 4], IVec4, uniform_4_i32);
impl_Uniformable!(&[i32], Int, uniform_1_i32_slice);
impl_Uniformable!(&[[i32; 2]], IVec2, uniform_2_i32_slice);
impl_Uniformable!(&[[i32; 3]], IVec3, uniform_3_i32_slice);
impl_Uniformable!(&[[i32; 4]], IVec4, uniform_4_i32_slice);

impl_Uniformable!(u32, UInt, uniform_1_u32);
impl_Uniformable!([u32; 2], UIVec2, uniform_2_u32);
impl_Uniformable!([u32; 3], UIVec3, uniform_3_u32);
impl_Uniformable!([u32; 4], UIVec4, uniform_4_u32);
impl_Uniformable!(&[u32], UInt, uniform_1_u32_slice);
impl_Uniformable!(&[[u32; 2]], UIVec2, uniform_2_u32_slice);
impl_Uniformable!(&[[u32; 3]], UIVec3, uniform_3_u32_slice);
impl_Uniformable!(&[[u32; 4]], UIVec4, uniform_4_u32_slice);

impl_Uniformable!(f32, Float, uniform_1_f32);
impl_Uniformable!([f32; 2], Vec2, uniform_2_f32);
impl_Uniformable!([f32; 3], Vec3, uniform_3_f32);
impl_Uniformable!([f32; 4], Vec4, uniform_4_f32);
impl_Uniformable!(&[f32], Float, uniform_1_f32_slice);
impl_Uniformable!(&[[f32; 2]], Vec2, uniform_2_f32_slice);
impl_Uniformable!(&[[f32; 3]], Vec3, uniform_3_f32_slice);
impl_Uniformable!(&[[f32; 4]], Vec4, uniform_4_f32_slice);

// please don’t judge me
impl_Uniformable!(mat f32; 2, M22, uniform_matrix_2_f32_slice);
impl_Uniformable!(mat & f32; 2, M22, uniform_matrix_2_f32_slice);

impl_Uniformable!(mat f32; 3, M33, uniform_matrix_3_f32_slice);
impl_Uniformable!(mat & f32; 3, M33, uniform_matrix_3_f32_slice);

impl_Uniformable!(mat f32; 4, M44, uniform_matrix_4_f32_slice);
impl_Uniformable!(mat & f32; 4, M44, uniform_matrix_4_f32_slice);

// Special exception for booleans: because we cannot simply send the bool Rust type down to the GPU,
// we have to convert them to 32-bit integer (unsigned), which is lame, but well, WebGL / OpenGL,
// whatcha wanna do.
unsafe impl Uniformable<Glow> for bool {
    unsafe fn ty() -> UniformType {
        UniformType::Bool
    }

    unsafe fn update(self, program: &mut Program, uniform: &Uniform<Self>) {
        program.state.borrow().ctx.uniform_1_u32(
            program.location_map.borrow().get(&uniform.index()),
            self as u32,
        );
    }
}

unsafe impl Uniformable<Glow> for [bool; 2] {
    unsafe fn ty() -> UniformType {
        UniformType::BVec2
    }

    unsafe fn update(self, program: &mut Program, uniform: &Uniform<Self>) {
        program.state.borrow().ctx.uniform_2_u32(
            program.location_map.borrow().get(&uniform.index()),
            self[0] as u32,
            self[1] as u32,
        );
    }
}

unsafe impl Uniformable<Glow> for [bool; 3] {
    unsafe fn ty() -> UniformType {
        UniformType::BVec3
    }

    unsafe fn update(self, program: &mut Program, uniform: &Uniform<Self>) {
        program.state.borrow().ctx.uniform_3_u32(
            program.location_map.borrow().get(&uniform.index()),
            self[0] as u32,
            self[1] as u32,
            self[2] as u32,
        );
    }
}

unsafe impl Uniformable<Glow> for [bool; 4] {
    unsafe fn ty() -> UniformType {
        UniformType::BVec4
    }

    unsafe fn update(self, program: &mut Program, uniform: &Uniform<Self>) {
        program.state.borrow().ctx.uniform_4_u32(
            program.location_map.borrow().get(&uniform.index()),
            self[0] as u32,
            self[1] as u32,
            self[2] as u32,
            self[3] as u32,
        );
    }
}

unsafe impl<'a> Uniformable<Glow> for &'a [bool] {
    unsafe fn ty() -> UniformType {
        UniformType::Bool
    }

    unsafe fn update(self, program: &mut Program, uniform: &Uniform<Self>) {
        let v: Vec<_> = self.iter().map(|x| *x as u32).collect();

        program
            .state
            .borrow()
            .ctx
            .uniform_1_u32_slice(program.location_map.borrow().get(&uniform.index()), &v);
    }
}

unsafe impl<'a> Uniformable<Glow> for &'a [[bool; 2]] {
    unsafe fn ty() -> UniformType {
        UniformType::BVec2
    }

    unsafe fn update(self, program: &mut Program, uniform: &Uniform<Self>) {
        let v: Vec<_> = self.iter().map(|x| [x[0] as u32, x[1] as u32]).collect();
        let data = flatten_slice!(v: u32, len = 2 * v.len());

        program
            .state
            .borrow()
            .ctx
            .uniform_2_u32_slice(program.location_map.borrow().get(&uniform.index()), data);
    }
}

unsafe impl<'a> Uniformable<Glow> for &'a [[bool; 3]] {
    unsafe fn ty() -> UniformType {
        UniformType::BVec3
    }

    unsafe fn update(self, program: &mut Program, uniform: &Uniform<Self>) {
        let v: Vec<_> = self
            .iter()
            .map(|x| [x[0] as u32, x[1] as u32, x[2] as u32])
            .collect();
        let data = flatten_slice!(v: u32, len = 3 * v.len());

        program
            .state
            .borrow()
            .ctx
            .uniform_3_u32_slice(program.location_map.borrow().get(&uniform.index()), data);
    }
}

unsafe impl<'a> Uniformable<Glow> for &'a [[bool; 4]] {
    unsafe fn ty() -> UniformType {
        UniformType::BVec4
    }

    unsafe fn update(self, program: &mut Program, uniform: &Uniform<Self>) {
        let v: Vec<_> = self
            .iter()
            .map(|x| [x[0] as u32, x[1] as u32, x[2] as u32, x[3] as u32])
            .collect();
        let data = flatten_slice!(v: u32, len = 4 * v.len());

        program
            .state
            .borrow()
            .ctx
            .uniform_4_u32_slice(program.location_map.borrow().get(&uniform.index()), data);
    }
}

unsafe impl<T> Uniformable<Glow> for BufferBinding<T> {
    unsafe fn ty() -> UniformType {
        UniformType::BufferBinding
    }

    unsafe fn update(self, program: &mut Program, uniform: &Uniform<Self>) {
        program.state.borrow().ctx.uniform_block_binding(
            program.handle,
            uniform.index() as u32,
            self.binding(),
        );
    }
}

unsafe impl<D, S> Uniformable<Glow> for TextureBinding<D, S>
where
    D: Dimensionable,
    S: SamplerType,
{
    unsafe fn ty() -> UniformType {
        match (S::sample_type(), D::dim()) {
            (PixelType::NormIntegral, Dim::Dim1) => UniformType::Sampler1D,
            (PixelType::NormUnsigned, Dim::Dim1) => UniformType::Sampler1D,
            (PixelType::Integral, Dim::Dim1) => UniformType::ISampler1D,
            (PixelType::Unsigned, Dim::Dim1) => UniformType::UISampler1D,
            (PixelType::Floating, Dim::Dim1) => UniformType::Sampler1D,

            (PixelType::NormIntegral, Dim::Dim2) => UniformType::Sampler2D,
            (PixelType::NormUnsigned, Dim::Dim2) => UniformType::Sampler2D,
            (PixelType::Integral, Dim::Dim2) => UniformType::ISampler2D,
            (PixelType::Unsigned, Dim::Dim2) => UniformType::UISampler2D,
            (PixelType::Floating, Dim::Dim2) => UniformType::Sampler2D,

            (PixelType::NormIntegral, Dim::Dim3) => UniformType::Sampler3D,
            (PixelType::NormUnsigned, Dim::Dim3) => UniformType::Sampler3D,
            (PixelType::Integral, Dim::Dim3) => UniformType::ISampler3D,
            (PixelType::Unsigned, Dim::Dim3) => UniformType::UISampler3D,
            (PixelType::Floating, Dim::Dim3) => UniformType::Sampler3D,

            (PixelType::NormIntegral, Dim::Cubemap) => UniformType::Cubemap,
            (PixelType::NormUnsigned, Dim::Cubemap) => UniformType::Cubemap,
            (PixelType::Integral, Dim::Cubemap) => UniformType::ICubemap,
            (PixelType::Unsigned, Dim::Cubemap) => UniformType::UICubemap,
            (PixelType::Floating, Dim::Cubemap) => UniformType::Cubemap,

            (PixelType::NormIntegral, Dim::Dim1Array) => UniformType::Sampler1DArray,
            (PixelType::NormUnsigned, Dim::Dim1Array) => UniformType::Sampler1DArray,
            (PixelType::Integral, Dim::Dim1Array) => UniformType::ISampler1DArray,
            (PixelType::Unsigned, Dim::Dim1Array) => UniformType::UISampler1DArray,
            (PixelType::Floating, Dim::Dim1Array) => UniformType::Sampler1DArray,

            (PixelType::NormIntegral, Dim::Dim2Array) => UniformType::Sampler2DArray,
            (PixelType::NormUnsigned, Dim::Dim2Array) => UniformType::Sampler2DArray,
            (PixelType::Integral, Dim::Dim2Array) => UniformType::ISampler2DArray,
            (PixelType::Unsigned, Dim::Dim2Array) => UniformType::UISampler2DArray,
            (PixelType::Floating, Dim::Dim2Array) => UniformType::Sampler2DArray,
        }
    }

    unsafe fn update(self, program: &mut Program, uniform: &Uniform<Self>) {
        program.state.borrow().ctx.uniform_1_i32(
            program.location_map.borrow().get(&uniform.index()),
            self.binding() as i32,
        );
    }
}
