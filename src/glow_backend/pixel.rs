use luminance::pixel::{Format, PixelFormat, Size, Type};
use web_sys::WebGl2RenderingContext;

// WebGL format, internal sized-format and type.
pub(crate) fn webgl_pixel_format(pf: PixelFormat) -> Option<(u32, u32, u32)> {
  match (pf.format, pf.encoding) {
    // red channel
    (Format::R(Size::Eight), Type::NormUnsigned) => Some((
      WebGl2RenderingContext::RED,
      WebGl2RenderingContext::R8,
      WebGl2RenderingContext::UNSIGNED_BYTE,
    )),
    (Format::R(Size::Eight), Type::NormIntegral) => Some((
      WebGl2RenderingContext::RED,
      WebGl2RenderingContext::R8_SNORM,
      WebGl2RenderingContext::BYTE,
    )),
    (Format::R(Size::Eight), Type::Integral) => Some((
      WebGl2RenderingContext::RED_INTEGER,
      WebGl2RenderingContext::R8I,
      WebGl2RenderingContext::BYTE,
    )),
    (Format::R(Size::Eight), Type::Unsigned) => Some((
      WebGl2RenderingContext::RED_INTEGER,
      WebGl2RenderingContext::R8UI,
      WebGl2RenderingContext::UNSIGNED_BYTE,
    )),

    (Format::R(Size::Sixteen), Type::Integral) => Some((
      WebGl2RenderingContext::RED_INTEGER,
      WebGl2RenderingContext::R16I,
      WebGl2RenderingContext::SHORT,
    )),
    (Format::R(Size::Sixteen), Type::Unsigned) => Some((
      WebGl2RenderingContext::RED_INTEGER,
      WebGl2RenderingContext::R16UI,
      WebGl2RenderingContext::UNSIGNED_SHORT,
    )),

    (Format::R(Size::ThirtyTwo), Type::NormUnsigned) => Some((
      WebGl2RenderingContext::RED_INTEGER,
      WebGl2RenderingContext::RED,
      WebGl2RenderingContext::UNSIGNED_INT,
    )),
    (Format::R(Size::ThirtyTwo), Type::NormIntegral) => Some((
      WebGl2RenderingContext::RED_INTEGER,
      WebGl2RenderingContext::RED,
      WebGl2RenderingContext::INT,
    )),
    (Format::R(Size::ThirtyTwo), Type::Integral) => Some((
      WebGl2RenderingContext::RED_INTEGER,
      WebGl2RenderingContext::R32I,
      WebGl2RenderingContext::INT,
    )),
    (Format::R(Size::ThirtyTwo), Type::Unsigned) => Some((
      WebGl2RenderingContext::RED_INTEGER,
      WebGl2RenderingContext::R32UI,
      WebGl2RenderingContext::UNSIGNED_INT,
    )),
    (Format::R(Size::ThirtyTwo), Type::Floating) => Some((
      WebGl2RenderingContext::RED,
      WebGl2RenderingContext::R32F,
      WebGl2RenderingContext::FLOAT,
    )),

    // red, blue channels
    (Format::RG(Size::Eight, Size::Eight), Type::NormUnsigned) => Some((
      WebGl2RenderingContext::RG,
      WebGl2RenderingContext::RG8,
      WebGl2RenderingContext::UNSIGNED_BYTE,
    )),
    (Format::RG(Size::Eight, Size::Eight), Type::NormIntegral) => Some((
      WebGl2RenderingContext::RG,
      WebGl2RenderingContext::RG8_SNORM,
      WebGl2RenderingContext::BYTE,
    )),
    (Format::RG(Size::Eight, Size::Eight), Type::Integral) => Some((
      WebGl2RenderingContext::RG_INTEGER,
      WebGl2RenderingContext::RG8I,
      WebGl2RenderingContext::BYTE,
    )),
    (Format::RG(Size::Eight, Size::Eight), Type::Unsigned) => Some((
      WebGl2RenderingContext::RG_INTEGER,
      WebGl2RenderingContext::RG8UI,
      WebGl2RenderingContext::UNSIGNED_BYTE,
    )),

    (Format::RG(Size::Sixteen, Size::Sixteen), Type::Integral) => Some((
      WebGl2RenderingContext::RG_INTEGER,
      WebGl2RenderingContext::RG16I,
      WebGl2RenderingContext::SHORT,
    )),
    (Format::RG(Size::Sixteen, Size::Sixteen), Type::Unsigned) => Some((
      WebGl2RenderingContext::RG_INTEGER,
      WebGl2RenderingContext::RG16UI,
      WebGl2RenderingContext::UNSIGNED_SHORT,
    )),

    (Format::RG(Size::ThirtyTwo, Size::ThirtyTwo), Type::NormUnsigned) => Some((
      WebGl2RenderingContext::RG,
      WebGl2RenderingContext::RG,
      WebGl2RenderingContext::UNSIGNED_INT,
    )),
    (Format::RG(Size::ThirtyTwo, Size::ThirtyTwo), Type::NormIntegral) => Some((
      WebGl2RenderingContext::RG,
      WebGl2RenderingContext::RG,
      WebGl2RenderingContext::INT,
    )),
    (Format::RG(Size::ThirtyTwo, Size::ThirtyTwo), Type::Integral) => Some((
      WebGl2RenderingContext::RG_INTEGER,
      WebGl2RenderingContext::RG32I,
      WebGl2RenderingContext::INT,
    )),
    (Format::RG(Size::ThirtyTwo, Size::ThirtyTwo), Type::Unsigned) => Some((
      WebGl2RenderingContext::RG_INTEGER,
      WebGl2RenderingContext::RG32UI,
      WebGl2RenderingContext::UNSIGNED_INT,
    )),
    (Format::RG(Size::ThirtyTwo, Size::ThirtyTwo), Type::Floating) => Some((
      WebGl2RenderingContext::RG,
      WebGl2RenderingContext::RG32F,
      WebGl2RenderingContext::FLOAT,
    )),

    // red, blue, green channels
    (Format::RGB(Size::Eight, Size::Eight, Size::Eight), Type::NormUnsigned) => Some((
      WebGl2RenderingContext::RGB,
      WebGl2RenderingContext::RGB8,
      WebGl2RenderingContext::UNSIGNED_BYTE,
    )),
    (Format::RGB(Size::Eight, Size::Eight, Size::Eight), Type::NormIntegral) => Some((
      WebGl2RenderingContext::RGB,
      WebGl2RenderingContext::RGB8_SNORM,
      WebGl2RenderingContext::BYTE,
    )),
    (Format::RGB(Size::Eight, Size::Eight, Size::Eight), Type::Integral) => Some((
      WebGl2RenderingContext::RGB_INTEGER,
      WebGl2RenderingContext::RGB8I,
      WebGl2RenderingContext::BYTE,
    )),
    (Format::RGB(Size::Eight, Size::Eight, Size::Eight), Type::Unsigned) => Some((
      WebGl2RenderingContext::RGB_INTEGER,
      WebGl2RenderingContext::RGB8UI,
      WebGl2RenderingContext::UNSIGNED_BYTE,
    )),

    (Format::RGB(Size::Sixteen, Size::Sixteen, Size::Sixteen), Type::Integral) => Some((
      WebGl2RenderingContext::RGB_INTEGER,
      WebGl2RenderingContext::RGB16I,
      WebGl2RenderingContext::SHORT,
    )),
    (Format::RGB(Size::Sixteen, Size::Sixteen, Size::Sixteen), Type::Unsigned) => Some((
      WebGl2RenderingContext::RGB_INTEGER,
      WebGl2RenderingContext::RGB16UI,
      WebGl2RenderingContext::UNSIGNED_SHORT,
    )),

    (Format::RGB(Size::Eleven, Size::Eleven, Size::Ten), Type::Floating) => Some((
      WebGl2RenderingContext::RGB,
      WebGl2RenderingContext::R11F_G11F_B10F,
      WebGl2RenderingContext::FLOAT,
    )),

    (Format::RGB(Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo), Type::NormUnsigned) => Some((
      WebGl2RenderingContext::RGB,
      WebGl2RenderingContext::RGB,
      WebGl2RenderingContext::UNSIGNED_INT,
    )),
    (Format::RGB(Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo), Type::NormIntegral) => Some((
      WebGl2RenderingContext::RGB,
      WebGl2RenderingContext::RGB,
      WebGl2RenderingContext::INT,
    )),
    (Format::RGB(Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo), Type::Integral) => Some((
      WebGl2RenderingContext::RGB_INTEGER,
      WebGl2RenderingContext::RGB32I,
      WebGl2RenderingContext::INT,
    )),
    (Format::RGB(Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo), Type::Unsigned) => Some((
      WebGl2RenderingContext::RGB_INTEGER,
      WebGl2RenderingContext::RGB32UI,
      WebGl2RenderingContext::UNSIGNED_INT,
    )),
    (Format::RGB(Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo), Type::Floating) => Some((
      WebGl2RenderingContext::RGB,
      WebGl2RenderingContext::RGB32F,
      WebGl2RenderingContext::FLOAT,
    )),

    // red, blue, green, alpha channels
    (Format::RGBA(Size::Eight, Size::Eight, Size::Eight, Size::Eight), Type::NormUnsigned) => {
      Some((
        WebGl2RenderingContext::RGBA,
        WebGl2RenderingContext::RGBA8,
        WebGl2RenderingContext::UNSIGNED_BYTE,
      ))
    }
    (Format::RGBA(Size::Eight, Size::Eight, Size::Eight, Size::Eight), Type::NormIntegral) => {
      Some((
        WebGl2RenderingContext::RGBA,
        WebGl2RenderingContext::RGBA8_SNORM,
        WebGl2RenderingContext::BYTE,
      ))
    }
    (Format::RGBA(Size::Eight, Size::Eight, Size::Eight, Size::Eight), Type::Integral) => Some((
      WebGl2RenderingContext::RGBA_INTEGER,
      WebGl2RenderingContext::RGBA8I,
      WebGl2RenderingContext::BYTE,
    )),
    (Format::RGBA(Size::Eight, Size::Eight, Size::Eight, Size::Eight), Type::Unsigned) => Some((
      WebGl2RenderingContext::RGBA_INTEGER,
      WebGl2RenderingContext::RGBA8UI,
      WebGl2RenderingContext::UNSIGNED_BYTE,
    )),

    (Format::RGBA(Size::Sixteen, Size::Sixteen, Size::Sixteen, Size::Sixteen), Type::Integral) => {
      Some((
        WebGl2RenderingContext::RGBA_INTEGER,
        WebGl2RenderingContext::RGBA16I,
        WebGl2RenderingContext::SHORT,
      ))
    }
    (Format::RGBA(Size::Sixteen, Size::Sixteen, Size::Sixteen, Size::Sixteen), Type::Unsigned) => {
      Some((
        WebGl2RenderingContext::RGBA_INTEGER,
        WebGl2RenderingContext::RGBA16UI,
        WebGl2RenderingContext::UNSIGNED_SHORT,
      ))
    }

    (
      Format::RGBA(Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo),
      Type::NormUnsigned,
    ) => Some((
      WebGl2RenderingContext::RGBA,
      WebGl2RenderingContext::RGBA,
      WebGl2RenderingContext::UNSIGNED_INT,
    )),
    (
      Format::RGBA(Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo),
      Type::NormIntegral,
    ) => Some((
      WebGl2RenderingContext::RGBA,
      WebGl2RenderingContext::RGBA,
      WebGl2RenderingContext::INT,
    )),
    (
      Format::RGBA(Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo),
      Type::Integral,
    ) => Some((
      WebGl2RenderingContext::RGBA_INTEGER,
      WebGl2RenderingContext::RGBA32I,
      WebGl2RenderingContext::INT,
    )),
    (
      Format::RGBA(Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo),
      Type::Unsigned,
    ) => Some((
      WebGl2RenderingContext::RGBA_INTEGER,
      WebGl2RenderingContext::RGBA32UI,
      WebGl2RenderingContext::UNSIGNED_INT,
    )),
    (
      Format::RGBA(Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo),
      Type::Floating,
    ) => Some((
      WebGl2RenderingContext::RGBA,
      WebGl2RenderingContext::RGBA32F,
      WebGl2RenderingContext::FLOAT,
    )),

    // sRGB
    (Format::SRGB(Size::Eight, Size::Eight, Size::Eight), Type::NormUnsigned) => Some((
      WebGl2RenderingContext::RGB,
      WebGl2RenderingContext::SRGB8,
      WebGl2RenderingContext::UNSIGNED_BYTE,
    )),
    (Format::SRGB(Size::Eight, Size::Eight, Size::Eight), Type::NormIntegral) => Some((
      WebGl2RenderingContext::RGB,
      WebGl2RenderingContext::SRGB8,
      WebGl2RenderingContext::BYTE,
    )),
    (Format::SRGBA(Size::Eight, Size::Eight, Size::Eight, Size::Eight), Type::NormUnsigned) => {
      Some((
        WebGl2RenderingContext::RGBA,
        WebGl2RenderingContext::SRGB8_ALPHA8,
        WebGl2RenderingContext::UNSIGNED_BYTE,
      ))
    }
    (Format::SRGBA(Size::Eight, Size::Eight, Size::Eight, Size::Eight), Type::NormIntegral) => {
      Some((
        WebGl2RenderingContext::RGBA,
        WebGl2RenderingContext::SRGB8_ALPHA8,
        WebGl2RenderingContext::BYTE,
      ))
    }

    (Format::Depth(Size::ThirtyTwo), Type::Floating) => Some((
      WebGl2RenderingContext::DEPTH_COMPONENT,
      WebGl2RenderingContext::DEPTH_COMPONENT32F,
      WebGl2RenderingContext::FLOAT,
    )),

    _ => None,
  }
}
