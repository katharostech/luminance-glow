//! Glow backend for Luminance
//!
//! This crate provides a [glow] backend for [luminance].
//!
//! [luminance]: https://crates.io/crates/luminance
//! [glow]: https://github.com/grovesNL/glow

use std::cell::RefCell;
use std::rc::Rc;

#[macro_use]
mod slice;

mod buffer;
mod framebuffer;
mod pipeline;
mod pixel;
mod shader;
mod state;
mod tess;
mod texture;

use glow::Context as GlowContext;

use state::GlowState;
pub use state::StateQueryError;

/// The GLSL shader version to use
///
/// This effects the version heading added automatically to the top of the shader strings provided
/// to luminance.
#[derive(Debug, Clone, Copy)]
pub enum ShaderVersion {
    Gles3,
    Gles1,
}

/// The graphics context which must be provided to create a [`Glow`] backend
pub struct Context {
    glow_context: GlowContext,
    is_webgl1: bool,
    shader_version: ShaderVersion,
}

impl Context {
    /// Create a native context from a GL loader function
    #[cfg(not(wasm))]
    pub unsafe fn from_loader_function<F>(loader_function: F, shader_version: ShaderVersion) -> Self
    where
        F: FnMut(&str) -> *const std::os::raw::c_void,
    {
        Self {
            glow_context: GlowContext::from_loader_function(loader_function),
            is_webgl1: false,
            shader_version,
        }
    }

    /// Create a WebGL 1 context
    ///
    /// > ⚠️ **Warning:** The WebGL 1 backend has limitations that the native and WebGL 2 bakcends
    /// > to not have. The exact limitations are outside of the scope of this note, but include
    /// > things like limited support for different pixel formats, etc.
    #[cfg(wasm)]
    pub fn from_webgl1_context(context: web_sys::WebGlRenderingContext) -> Self {
        Self {
            glow_context: GlowContext::from_webgl1_context(context),
            is_webgl1: true,
            shader_version: ShaderVersion::Gles1,
        }
    }

    /// Create a WebGL 2 context
    #[cfg(wasm)]
    pub fn from_webgl2_context(
        context: web_sys::WebGl2RenderingContext,
        shader_version: ShaderVersion,
    ) -> Self {
        Self {
            glow_context: GlowContext::from_webgl2_context(context),
            is_webgl1: false,
            shader_version,
        }
    }
}

/// The Luminance Glow backend
#[derive(Debug)]
pub struct Glow {
    pub(crate) state: Rc<RefCell<GlowState>>,
}

impl Glow {
    /// Create a glow backend instance from a `glow` [`Context`][glow::Context]
    pub fn from_context(ctx: Context) -> Result<Self, StateQueryError> {
        let Context {
            glow_context,
            is_webgl1,
            shader_version,
        } = ctx;
        GlowState::new(glow_context, is_webgl1, shader_version).map(|state| Glow {
            state: Rc::new(RefCell::new(state)),
        })
    }
}
