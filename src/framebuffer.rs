//! Framebuffer support for WebGL2.

use luminance::backend::color_slot::ColorSlot;
use luminance::backend::depth_slot::DepthSlot;
use luminance::backend::framebuffer::{Framebuffer as FramebufferBackend, FramebufferBackBuffer};
use luminance::framebuffer::{FramebufferError, IncompleteReason};
use luminance::texture::{Dim2, Dimensionable, Sampler};
use std::cell::RefCell;
use std::rc::Rc;

use crate::state::GlowState;
use crate::GlowBackend;
use glow::HasContext;

pub struct Framebuffer<D>
where
    D: Dimensionable,
{
    // None is the default framebuffer…
    pub(crate) handle: Option<glow::Framebuffer>,
    renderbuffer: Option<glow::Renderbuffer>,
    pub(crate) size: D::Size,
    state: Rc<RefCell<GlowState>>,
}

impl<D> Drop for Framebuffer<D>
where
    D: Dimensionable,
{
    fn drop(&mut self) {
        unsafe {
            let state = self.state.borrow();

            self.renderbuffer.map(|x| state.ctx.delete_renderbuffer(x));
            self.handle.map(|x| state.ctx.delete_framebuffer(x));
        }
    }
}

unsafe impl<D> FramebufferBackend<D> for GlowBackend
where
    D: Dimensionable,
{
    type FramebufferRepr = Framebuffer<D>;

    unsafe fn new_framebuffer<CS, DS>(
        &mut self,
        size: D::Size,
        _: usize,
        _: &Sampler,
    ) -> Result<Self::FramebufferRepr, FramebufferError>
    where
        CS: ColorSlot<Self, D>,
        DS: DepthSlot<Self, D>,
    {
        let color_formats = CS::color_formats();
        let depth_format = DS::depth_format();
        let mut depth_renderbuffer = None;

        let mut state = self.state.borrow_mut();

        let handle = state
            .create_framebuffer()
            .map_err(|_| FramebufferError::cannot_create())?;
        state.bind_draw_framebuffer(Some(handle));

        // reserve textures to speed up slots creation
        let textures_needed = color_formats.len() + depth_format.map_or(0, |_| 1);
        state.reserve_textures(textures_needed);

        // color textures
        if color_formats.is_empty() {
            state.ctx.draw_buffers(&[glow::NONE]);
        } else {
            // Specify the list of color buffers to draw to; to do so, we need to generate a temporary
            // list (Vec) of 32-bit integers and turn it into a Uint32Array to pass it across WASM
            // boundary.
            let color_buf_nb = color_formats.len() as u32;
            let color_buffers: Vec<_> =
                (glow::COLOR_ATTACHMENT0..glow::COLOR_ATTACHMENT0 + color_buf_nb).collect();

            state.ctx.draw_buffers(color_buffers.as_ref());
        }

        // depth texture
        if depth_format.is_none() {
            let renderbuffer = state
                .ctx
                .create_renderbuffer()
                .map_err(|_| FramebufferError::cannot_create())?;

            state
                .ctx
                .bind_renderbuffer(glow::RENDERBUFFER, Some(renderbuffer));

            state.ctx.renderbuffer_storage(
                glow::RENDERBUFFER,
                glow::DEPTH_COMPONENT32F,
                D::width(size) as i32,
                D::height(size) as i32,
            );
            state.ctx.framebuffer_renderbuffer(
                glow::FRAMEBUFFER,
                glow::DEPTH_ATTACHMENT,
                glow::RENDERBUFFER,
                Some(renderbuffer),
            );

            depth_renderbuffer = Some(renderbuffer);
        }

        let framebuffer = Framebuffer {
            handle: Some(handle),
            renderbuffer: depth_renderbuffer,
            size,
            state: self.state.clone(),
        };

        Ok(framebuffer)
    }

    unsafe fn attach_color_texture(
        framebuffer: &mut Self::FramebufferRepr,
        texture: &Self::TextureRepr,
        attachment_index: usize,
    ) -> Result<(), FramebufferError> {
        match texture.target {
            glow::TEXTURE_2D => {
                let state = framebuffer.state.borrow();
                state.ctx.framebuffer_texture_2d(
                    glow::FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0 + attachment_index as u32,
                    texture.target,
                    Some(texture.handle),
                    0,
                );

                Ok(())
            }

            _ => Err(FramebufferError::unsupported_attachment()),
        }
    }

    unsafe fn attach_depth_texture(
        framebuffer: &mut Self::FramebufferRepr,
        texture: &Self::TextureRepr,
    ) -> Result<(), FramebufferError> {
        match texture.target {
            glow::TEXTURE_2D => {
                let state = framebuffer.state.borrow();
                state.ctx.framebuffer_texture_2d(
                    glow::FRAMEBUFFER,
                    glow::DEPTH_ATTACHMENT,
                    texture.target,
                    Some(texture.handle),
                    0,
                );

                Ok(())
            }

            _ => Err(FramebufferError::unsupported_attachment()),
        }
    }

    unsafe fn validate_framebuffer(
        framebuffer: Self::FramebufferRepr,
    ) -> Result<Self::FramebufferRepr, FramebufferError> {
        get_framebuffer_status(&mut framebuffer.state.borrow_mut())?;
        Ok(framebuffer)
    }

    unsafe fn framebuffer_size(framebuffer: &Self::FramebufferRepr) -> D::Size {
        framebuffer.size
    }
}

fn get_framebuffer_status(state: &mut GlowState) -> Result<(), IncompleteReason> {
    unsafe {
        let status = state.ctx.check_framebuffer_status(glow::FRAMEBUFFER);

        match status {
            glow::FRAMEBUFFER_COMPLETE => Ok(()),
            glow::FRAMEBUFFER_INCOMPLETE_ATTACHMENT => Err(IncompleteReason::IncompleteAttachment),
            glow::FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT => {
                Err(IncompleteReason::MissingAttachment)
            }
            glow::FRAMEBUFFER_UNSUPPORTED => Err(IncompleteReason::Unsupported),
            glow::FRAMEBUFFER_INCOMPLETE_MULTISAMPLE => {
                Err(IncompleteReason::IncompleteMultisample)
            }
            _ => panic!(
                "unknown WebGL2 framebuffer incomplete status! status={}",
                status
            ),
        }
    }
}

unsafe impl FramebufferBackBuffer for GlowBackend {
    unsafe fn back_buffer(
        &mut self,
        size: <Dim2 as Dimensionable>::Size,
    ) -> Result<Self::FramebufferRepr, FramebufferError> {
        Ok(Framebuffer {
            handle: None, // None is the default framebuffer in WebGL
            renderbuffer: None,
            size,
            state: self.state.clone(),
        })
    }
}
