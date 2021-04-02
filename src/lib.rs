//! Glow backend for Luminance
//!
//! This crate provides a [luminance] backend for [glow].
//!
//! [luminance]: https://crates.io/crates/luminance
//! [glow]: https://github.com/grovesNL/glow

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

use crate::state::GlowState;
pub use crate::state::StateQueryError;
use std::cell::RefCell;
use std::rc::Rc;

/// The Glow backend.
#[derive(Debug)]
pub struct Glow {
    pub(crate) state: Rc<RefCell<GlowState>>,
}

impl Glow {
    pub fn from_context(ctx: glow::Context) -> Result<Self, StateQueryError> {
        GlowState::new(ctx).map(|state| Glow {
            state: Rc::new(RefCell::new(state)),
        })
    }
}
