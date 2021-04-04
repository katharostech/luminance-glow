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

/// The graphics context which must be provided to create a [`Glow`] backend
pub struct Context {
    glow_context: GlowContext,
    is_webgl1: bool,
}

impl Context {
    /// Create a native context from a GL loader function
    #[cfg(not(wasm))]
    pub unsafe fn from_loader_function<F>(mut loader_function: F) -> Self
    where
        F: FnMut(&str) -> *const std::os::raw::c_void,
    {
        Self {
            glow_context: GlowContext::from_loader_function(loader_function),
            is_webgl1: false,
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
        }
    }

    /// Create a WebGL 2 context
    #[cfg(wasm)]
    pub fn from_webgl2_context(context: web_sys::WebGl2RenderingContext) -> Self {
        Self {
            glow_context: GlowContext::from_webgl2_context(context),
            is_webgl1: false,
        }
    }
}

/// The Luminance Glow backend
#[derive(Debug)]
pub struct Glow {
    pub(crate) state: Rc<RefCell<GlowState>>,
    pub(crate) is_webgl1: bool,
}

impl Glow {
    /// Create a glow backend instance from a `glow` [`Context`][glow::Context]
    pub fn from_context(ctx: Context) -> Result<Self, StateQueryError> {
        let Context {
            glow_context,
            is_webgl1,
        } = ctx;
        GlowState::new(glow_context, is_webgl1).map(|state| Glow {
            state: Rc::new(RefCell::new(state)),
            is_webgl1,
        })
    }
}
