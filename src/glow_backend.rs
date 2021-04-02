//! Glow Backend support

pub mod buffer;
pub mod framebuffer;
pub mod pipeline;
pub mod pixel;
pub mod shader;
pub mod state;
pub mod tess;
pub mod texture;

use crate::glow_backend::state::GlowState;
pub use crate::glow_backend::state::StateQueryError;
use std::cell::RefCell;
use std::rc::Rc;

/// The WebGL2 backend.
#[derive(Debug)]
pub struct GlowBackend {
    pub(crate) state: Rc<RefCell<GlowState>>,
}

impl GlowBackend {
    pub fn new(ctx: glow::Context) -> Result<Self, StateQueryError> {
        GlowState::new(ctx).map(|state| GlowBackend {
            state: Rc::new(RefCell::new(state)),
        })
    }
}
