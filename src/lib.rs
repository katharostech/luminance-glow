//! Glow backend for Luminance
//!
//! This crate provides a [luminance] backend for [glow].
//!
//! [luminance]: https://crates.io/crates/luminance
//! [glow]: https://github.com/grovesNL/glow

extern crate serde_derive;

pub use glow;

#[macro_use]
mod slice;

pub mod glow_backend;

pub use glow_backend::GlowBackend;
