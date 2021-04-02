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

/// The glow graphics context which must be provided to create a [`Glow`] backend
pub use glow::Context;

use state::GlowState;
pub use state::StateQueryError;

/// The Luminance Glow backend
#[derive(Debug)]
pub struct Glow {
    pub(crate) state: Rc<RefCell<GlowState>>,
}

impl Glow {
    /// Create a glow backend instance from a `glow` [`Context`][glow::Context]
    pub fn from_context(ctx: Context) -> Result<Self, StateQueryError> {
        GlowState::new(ctx).map(|state| Glow {
            state: Rc::new(RefCell::new(state)),
        })
    }
}
