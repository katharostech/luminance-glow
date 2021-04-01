# Changelog

This document is the changelog of [luminance-webgl](https://crates.io/crates/luminance-webgl).
You should consult it when upgrading to a new version, as it contains precious information on
breaking changes, minor additions and patch notes.

**If you’re experiencing weird type errors when upgrading to a new version**, it might be due to
how `cargo` resolves dependencies. `cargo update` is not enough, because all luminance crate use
[SemVer ranges](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html) to stay
compatible with as many crates as possible. In that case, you want `cargo update --aggressive`.

<!-- vim-markdown-toc GFM -->

* [0.3.2](#032)
* [0.3.1](#031)
* [0.3](#03)
  * [Patch](#patch)
  * [Breaking changes](#breaking-changes)
* [0.2.1](#021)
* [0.2](#02)
* [0.1.2](#012)
* [0.1.1](#011)
* [0.1](#01)

<!-- vim-markdown-toc -->

# 0.3.2

> Oct 31st, 2020

- Fix a bug while getting the context’s initial value for the _depth write_ property (that one can change with
  `RenderState::set_depth_write`).

# 0.3.1

> Oct 31st, 2020

- Fix several uniform updates methods, that wouldn’t send data with the correct size, causing various random issues.

# 0.3

> Oct 28, 2020

## Patch

- Remove the limitation about creating contexts in WebGL: it is now possible to create as many as users want. WebGL
  doesn’t have the same requirements as OpenGL in terms of threading and contexts.

## Breaking changes

- Remove the `obtain_slice` and `obtain_slice_mut` methods. If you were using them, please feel free to use the `Deref`
  and `DerefMut` interface instead. It prevents one extra layer of useless validation via `Result`, since backends will
  simply always return `Ok(slice)`. The validation process is done when accessing the slice, e.g. `Buffer::slice` and
  `Buffer::slice_mut`.

# 0.2.1

> Oct 26th, 2020

- Add a bunch of `Debug` annotations.
- Add support for _scissor test_ implementation.

# 0.2

> Aug 30th, 2020

- Support of `luminance-0.42`.
- Add support for `UniformWarning::UnsupportedType`, which is raised when a uniform type is used by the client
  code while not supported by the backend implementation.

# 0.1.2

> Aug 18th, 2020

- Remove unnecessary type-erasure that was basically doing a no-op.
- Fix deinterleaved tessellation mapping that would map mutable slices with the wrong length.

# 0.1.1

> Jul 24th, 2020

- Support of `luminance-0.41`.

# 0.1

> Wed Jul, 15th 2020

- Initial revision.

[luminance-webgl]: https://crates.io/crates/luminance-webgl
