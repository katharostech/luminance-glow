//! Pipeline support for Glow.

use luminance::backend::pipeline::{
    Pipeline as PipelineBackend, PipelineBase, PipelineBuffer, PipelineTexture,
};
use luminance::backend::render_gate::RenderGate;
use luminance::backend::shading_gate::ShadingGate;
use luminance::backend::tess::Tess;
use luminance::backend::tess_gate::TessGate;
use luminance::blending::BlendingMode;
use luminance::pipeline::{PipelineError, PipelineState, Viewport};
use luminance::pixel::Pixel;
use luminance::render_state::RenderState;
use luminance::tess::{Deinterleaved, DeinterleavedData, Interleaved, TessIndex, TessVertexData};
use luminance::texture::Dimensionable;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::{
    state::{BlendingState, DepthTest, FaceCullingState, GlowState, ScissorState},
    Glow,
};

use glow::HasContext;

pub struct Pipeline {
    state: Rc<RefCell<GlowState>>,
}

pub struct BoundBuffer {
    pub(crate) binding: u32,
    state: Rc<RefCell<GlowState>>,
}

impl Drop for BoundBuffer {
    fn drop(&mut self) {
        // place the binding into the free list
        self.state
            .borrow_mut()
            .binding_stack_mut()
            .free_buffer_bindings
            .push(self.binding);
    }
}

pub struct BoundTexture<D, P>
where
    D: Dimensionable,
    P: Pixel,
{
    pub(crate) unit: u32,
    state: Rc<RefCell<GlowState>>,
    _phantom: PhantomData<*const (D, P)>,
}

impl<D, P> Drop for BoundTexture<D, P>
where
    D: Dimensionable,
    P: Pixel,
{
    fn drop(&mut self) {
        // place the binding into the free list
        self.state
            .borrow_mut()
            .binding_stack_mut()
            .free_texture_units
            .push(self.unit);
    }
}

unsafe impl PipelineBase for Glow {
    type PipelineRepr = Pipeline;

    unsafe fn new_pipeline(&mut self) -> Result<Self::PipelineRepr, PipelineError> {
        let pipeline = Pipeline {
            state: self.state.clone(),
        };

        Ok(pipeline)
    }
}

unsafe impl<D> PipelineBackend<D> for Glow
where
    D: Dimensionable,
{
    unsafe fn start_pipeline(
        &mut self,
        framebuffer: &Self::FramebufferRepr,
        pipeline_state: &PipelineState,
    ) {
        let mut state = self.state.borrow_mut();

        state.bind_draw_framebuffer(framebuffer.handle);

        let clear_color = pipeline_state.clear_color;
        state.set_clear_color(clear_color);

        let size = framebuffer.size;

        let (x, y, w, h) = match pipeline_state.viewport {
            Viewport::Whole => (0, 0, D::width(size), D::height(size)),
            Viewport::Specific {
                x,
                y,
                width,
                height,
            } => (x, y, width, height),
        };

        state.set_viewport([x as _, y as _, w as _, h as _]);

        if pipeline_state.clear_color_enabled || pipeline_state.clear_depth_enabled {
            let color_bit = if pipeline_state.clear_color_enabled {
                glow::COLOR_BUFFER_BIT
            } else {
                0
            };

            let depth_bit = if pipeline_state.clear_depth_enabled {
                glow::DEPTH_BUFFER_BIT
            } else {
                0
            };

            // scissor test
            match pipeline_state.scissor() {
                Some(region) => {
                    state.set_scissor_state(ScissorState::On);
                    state.set_scissor_region(region);
                }

                None => {
                    state.set_scissor_state(ScissorState::Off);
                }
            }

            state.ctx.clear(color_bit | depth_bit);
        }
    }
}

unsafe impl<T> PipelineBuffer<T> for Glow
where
    T: Copy,
{
    type BoundBufferRepr = BoundBuffer;

    unsafe fn bind_buffer(
        pipeline: &Self::PipelineRepr,
        buffer: &Self::BufferRepr,
    ) -> Result<Self::BoundBufferRepr, PipelineError> {
        let mut state = pipeline.state.borrow_mut();
        let bstack = state.binding_stack_mut();

        let binding = bstack.free_buffer_bindings.pop().unwrap_or_else(|| {
            // no more free bindings; reserve one
            let binding = bstack.next_buffer_binding;
            bstack.next_buffer_binding += 1;
            binding
        });

        state.bind_buffer_base(buffer.handle(), binding);

        Ok(BoundBuffer {
            binding,
            state: pipeline.state.clone(),
        })
    }

    unsafe fn buffer_binding(bound: &Self::BoundBufferRepr) -> u32 {
        bound.binding
    }
}

unsafe impl<D, P> PipelineTexture<D, P> for Glow
where
    D: Dimensionable,
    P: Pixel,
{
    type BoundTextureRepr = BoundTexture<D, P>;

    unsafe fn bind_texture(
        pipeline: &Self::PipelineRepr,
        texture: &Self::TextureRepr,
    ) -> Result<Self::BoundTextureRepr, PipelineError>
    where
        D: Dimensionable,
        P: Pixel,
    {
        let mut state = pipeline.state.borrow_mut();
        let bstack = state.binding_stack_mut();

        let unit = bstack.free_texture_units.pop().unwrap_or_else(|| {
            // no more free units; reserve one
            let unit = bstack.next_texture_unit;
            bstack.next_texture_unit += 1;
            unit
        });

        state.set_texture_unit(unit);
        state.bind_texture(texture.target, Some(texture.handle()));

        Ok(BoundTexture {
            unit,
            state: pipeline.state.clone(),
            _phantom: PhantomData,
        })
    }

    unsafe fn texture_binding(bound: &Self::BoundTextureRepr) -> u32 {
        bound.unit
    }
}

unsafe impl<V, I, W> TessGate<V, I, W, Interleaved> for Glow
where
    V: TessVertexData<Interleaved, Data = Vec<V>>,
    I: TessIndex,
    W: TessVertexData<Interleaved, Data = Vec<W>>,
{
    unsafe fn render(
        &mut self,
        tess: &Self::TessRepr,
        start_index: usize,
        vert_nb: usize,
        inst_nb: usize,
    ) {
        let _ = <Self as Tess<V, I, W, Interleaved>>::render(tess, start_index, vert_nb, inst_nb);
    }
}

unsafe impl<V, I, W> TessGate<V, I, W, Deinterleaved> for Glow
where
    V: TessVertexData<Deinterleaved, Data = Vec<DeinterleavedData>>,
    I: TessIndex,
    W: TessVertexData<Deinterleaved, Data = Vec<DeinterleavedData>>,
{
    unsafe fn render(
        &mut self,
        tess: &Self::TessRepr,
        start_index: usize,
        vert_nb: usize,
        inst_nb: usize,
    ) {
        let _ = <Self as Tess<V, I, W, Deinterleaved>>::render(tess, start_index, vert_nb, inst_nb);
    }
}

unsafe impl RenderGate for Glow {
    unsafe fn enter_render_state(&mut self, rdr_st: &RenderState) {
        let mut state = self.state.borrow_mut();

        // blending state
        match rdr_st.blending() {
            Some(blending) => {
                state.set_blending_state(BlendingState::On);
                match blending {
                    BlendingMode::Combined(b) => {
                        state.set_blending_equation(b.equation);
                        state.set_blending_func(b.src, b.dst);
                    }
                    BlendingMode::Separate { rgb, alpha } => {
                        state.set_blending_equation_separate(rgb.equation, alpha.equation);
                        state.set_blending_func_separate(rgb.src, rgb.dst, alpha.src, alpha.dst);
                    }
                }
            }

            None => {
                state.set_blending_state(BlendingState::Off);
            }
        }

        // depth-related state
        if let Some(depth_comparison) = rdr_st.depth_test() {
            state.set_depth_test(DepthTest::On);
            state.set_depth_test_comparison(depth_comparison);
        } else {
            state.set_depth_test(DepthTest::Off);
        }

        state.set_depth_write(rdr_st.depth_write());

        // face culling state
        match rdr_st.face_culling() {
            Some(face_culling) => {
                state.set_face_culling_state(FaceCullingState::On);
                state.set_face_culling_order(face_culling.order);
                state.set_face_culling_mode(face_culling.mode);
            }
            None => {
                state.set_face_culling_state(FaceCullingState::Off);
            }
        }

        // scissor test
        match rdr_st.scissor() {
            Some(region) => {
                state.set_scissor_state(ScissorState::On);
                state.set_scissor_region(region);
            }

            None => {
                state.set_scissor_state(ScissorState::Off);
            }
        }
    }
}

unsafe impl ShadingGate for Glow {
    unsafe fn apply_shader_program(&mut self, shader_program: &Self::ProgramRepr) {
        self.state
            .borrow_mut()
            .use_program(Some(shader_program.handle));
    }
}
