//! Glow backend for Luminance
//!
//! This crate provides a [luminance] backend for [glow].
//!
//! [luminance]: https://crates.io/crates/luminance
//! [glow]: https://github.com/grovesNL/glow

use std::cell::RefCell;
use std::rc::Rc;

#[macro_use]
mod slice;

pub mod buffer;
pub mod framebuffer;
pub mod pipeline;
pub mod pixel;
pub mod shader;
pub mod state;
pub mod tess;
pub mod texture;

pub use glow::Context;

use state::{GlowState, StateQueryError};

/// The Glow backend.
#[derive(Debug)]
pub struct Glow {
    pub(crate) state: Rc<RefCell<GlowState>>,
}

impl Glow {
    /// Create a glow backend instance from a `glow` [`Context`][glow::Context].
    pub fn from_context(ctx: Context) -> Result<Self, StateQueryError> {
        GlowState::new(ctx).map(|state| Glow {
            state: Rc::new(RefCell::new(state)),
        })
    }
}
