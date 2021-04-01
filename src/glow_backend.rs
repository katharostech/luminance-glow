//! Glow Backend support

mod array_buffer;
pub mod buffer;
pub mod framebuffer;
pub mod pipeline;
pub mod pixel;
pub mod shader;
pub mod state;
pub mod tess;
pub mod texture;

pub use crate::glow_backend::array_buffer::IntoArrayBuffer;
pub use crate::glow_backend::state::StateQueryError;
use crate::glow_backend::state::GlowState;
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
