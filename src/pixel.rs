use luminance::pixel::{Format, PixelFormat, Size, Type};

// Glow format, internal sized-format and type.
pub(crate) fn glow_pixel_format(pf: PixelFormat) -> Option<(u32, u32, u32)> {
    match (pf.format, pf.encoding) {
        // red channel
        (Format::R(Size::Eight), Type::NormUnsigned) => {
            Some((glow::RED, glow::R8, glow::UNSIGNED_BYTE))
        }
        (Format::R(Size::Eight), Type::NormIntegral) => {
            Some((glow::RED, glow::R8_SNORM, glow::BYTE))
        }
        (Format::R(Size::Eight), Type::Integral) => {
            Some((glow::RED_INTEGER, glow::R8I, glow::BYTE))
        }
        (Format::R(Size::Eight), Type::Unsigned) => {
            Some((glow::RED_INTEGER, glow::R8UI, glow::UNSIGNED_BYTE))
        }

        (Format::R(Size::Sixteen), Type::Integral) => {
            Some((glow::RED_INTEGER, glow::R16I, glow::SHORT))
        }
        (Format::R(Size::Sixteen), Type::Unsigned) => {
            Some((glow::RED_INTEGER, glow::R16UI, glow::UNSIGNED_SHORT))
        }

        (Format::R(Size::ThirtyTwo), Type::NormUnsigned) => {
            Some((glow::RED_INTEGER, glow::RED, glow::UNSIGNED_INT))
        }
        (Format::R(Size::ThirtyTwo), Type::NormIntegral) => {
            Some((glow::RED_INTEGER, glow::RED, glow::INT))
        }
        (Format::R(Size::ThirtyTwo), Type::Integral) => {
            Some((glow::RED_INTEGER, glow::R32I, glow::INT))
        }
        (Format::R(Size::ThirtyTwo), Type::Unsigned) => {
            Some((glow::RED_INTEGER, glow::R32UI, glow::UNSIGNED_INT))
        }
        (Format::R(Size::ThirtyTwo), Type::Floating) => Some((glow::RED, glow::R32F, glow::FLOAT)),

        // red, blue channels
        (Format::RG(Size::Eight, Size::Eight), Type::NormUnsigned) => {
            Some((glow::RG, glow::RG8, glow::UNSIGNED_BYTE))
        }
        (Format::RG(Size::Eight, Size::Eight), Type::NormIntegral) => {
            Some((glow::RG, glow::RG8_SNORM, glow::BYTE))
        }
        (Format::RG(Size::Eight, Size::Eight), Type::Integral) => {
            Some((glow::RG_INTEGER, glow::RG8I, glow::BYTE))
        }
        (Format::RG(Size::Eight, Size::Eight), Type::Unsigned) => {
            Some((glow::RG_INTEGER, glow::RG8UI, glow::UNSIGNED_BYTE))
        }

        (Format::RG(Size::Sixteen, Size::Sixteen), Type::Integral) => {
            Some((glow::RG_INTEGER, glow::RG16I, glow::SHORT))
        }
        (Format::RG(Size::Sixteen, Size::Sixteen), Type::Unsigned) => {
            Some((glow::RG_INTEGER, glow::RG16UI, glow::UNSIGNED_SHORT))
        }

        (Format::RG(Size::ThirtyTwo, Size::ThirtyTwo), Type::NormUnsigned) => {
            Some((glow::RG, glow::RG, glow::UNSIGNED_INT))
        }
        (Format::RG(Size::ThirtyTwo, Size::ThirtyTwo), Type::NormIntegral) => {
            Some((glow::RG, glow::RG, glow::INT))
        }
        (Format::RG(Size::ThirtyTwo, Size::ThirtyTwo), Type::Integral) => {
            Some((glow::RG_INTEGER, glow::RG32I, glow::INT))
        }
        (Format::RG(Size::ThirtyTwo, Size::ThirtyTwo), Type::Unsigned) => {
            Some((glow::RG_INTEGER, glow::RG32UI, glow::UNSIGNED_INT))
        }
        (Format::RG(Size::ThirtyTwo, Size::ThirtyTwo), Type::Floating) => {
            Some((glow::RG, glow::RG32F, glow::FLOAT))
        }

        // red, blue, green channels
        (Format::RGB(Size::Eight, Size::Eight, Size::Eight), Type::NormUnsigned) => {
            Some((glow::RGB, glow::RGB8, glow::UNSIGNED_BYTE))
        }
        (Format::RGB(Size::Eight, Size::Eight, Size::Eight), Type::NormIntegral) => {
            Some((glow::RGB, glow::RGB8_SNORM, glow::BYTE))
        }
        (Format::RGB(Size::Eight, Size::Eight, Size::Eight), Type::Integral) => {
            Some((glow::RGB_INTEGER, glow::RGB8I, glow::BYTE))
        }
        (Format::RGB(Size::Eight, Size::Eight, Size::Eight), Type::Unsigned) => {
            Some((glow::RGB_INTEGER, glow::RGB8UI, glow::UNSIGNED_BYTE))
        }

        (Format::RGB(Size::Sixteen, Size::Sixteen, Size::Sixteen), Type::Integral) => {
            Some((glow::RGB_INTEGER, glow::RGB16I, glow::SHORT))
        }
        (Format::RGB(Size::Sixteen, Size::Sixteen, Size::Sixteen), Type::Unsigned) => {
            Some((glow::RGB_INTEGER, glow::RGB16UI, glow::UNSIGNED_SHORT))
        }

        (Format::RGB(Size::Eleven, Size::Eleven, Size::Ten), Type::Floating) => {
            Some((glow::RGB, glow::R11F_G11F_B10F, glow::FLOAT))
        }

        (Format::RGB(Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo), Type::NormUnsigned) => {
            Some((glow::RGB, glow::RGB, glow::UNSIGNED_INT))
        }
        (Format::RGB(Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo), Type::NormIntegral) => {
            Some((glow::RGB, glow::RGB, glow::INT))
        }
        (Format::RGB(Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo), Type::Integral) => {
            Some((glow::RGB_INTEGER, glow::RGB32I, glow::INT))
        }
        (Format::RGB(Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo), Type::Unsigned) => {
            Some((glow::RGB_INTEGER, glow::RGB32UI, glow::UNSIGNED_INT))
        }
        (Format::RGB(Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo), Type::Floating) => {
            Some((glow::RGB, glow::RGB32F, glow::FLOAT))
        }

        // red, blue, green, alpha channels
        (Format::RGBA(Size::Eight, Size::Eight, Size::Eight, Size::Eight), Type::NormUnsigned) => {
            Some((glow::RGBA, glow::RGBA8, glow::UNSIGNED_BYTE))
        }
        (Format::RGBA(Size::Eight, Size::Eight, Size::Eight, Size::Eight), Type::NormIntegral) => {
            Some((glow::RGBA, glow::RGBA8_SNORM, glow::BYTE))
        }
        (Format::RGBA(Size::Eight, Size::Eight, Size::Eight, Size::Eight), Type::Integral) => {
            Some((glow::RGBA_INTEGER, glow::RGBA8I, glow::BYTE))
        }
        (Format::RGBA(Size::Eight, Size::Eight, Size::Eight, Size::Eight), Type::Unsigned) => {
            Some((glow::RGBA_INTEGER, glow::RGBA8UI, glow::UNSIGNED_BYTE))
        }

        (
            Format::RGBA(Size::Sixteen, Size::Sixteen, Size::Sixteen, Size::Sixteen),
            Type::Integral,
        ) => Some((glow::RGBA_INTEGER, glow::RGBA16I, glow::SHORT)),
        (
            Format::RGBA(Size::Sixteen, Size::Sixteen, Size::Sixteen, Size::Sixteen),
            Type::Unsigned,
        ) => Some((glow::RGBA_INTEGER, glow::RGBA16UI, glow::UNSIGNED_SHORT)),

        (
            Format::RGBA(Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo),
            Type::NormUnsigned,
        ) => Some((glow::RGBA, glow::RGBA, glow::UNSIGNED_INT)),
        (
            Format::RGBA(Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo),
            Type::NormIntegral,
        ) => Some((glow::RGBA, glow::RGBA, glow::INT)),
        (
            Format::RGBA(Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo),
            Type::Integral,
        ) => Some((glow::RGBA_INTEGER, glow::RGBA32I, glow::INT)),
        (
            Format::RGBA(Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo),
            Type::Unsigned,
        ) => Some((glow::RGBA_INTEGER, glow::RGBA32UI, glow::UNSIGNED_INT)),
        (
            Format::RGBA(Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo, Size::ThirtyTwo),
            Type::Floating,
        ) => Some((glow::RGBA, glow::RGBA32F, glow::FLOAT)),

        // sRGB
        (Format::SRGB(Size::Eight, Size::Eight, Size::Eight), Type::NormUnsigned) => {
            Some((glow::RGB, glow::SRGB8, glow::UNSIGNED_BYTE))
        }
        (Format::SRGB(Size::Eight, Size::Eight, Size::Eight), Type::NormIntegral) => {
            Some((glow::RGB, glow::SRGB8, glow::BYTE))
        }
        (Format::SRGBA(Size::Eight, Size::Eight, Size::Eight, Size::Eight), Type::NormUnsigned) => {
            Some((glow::RGBA, glow::SRGB8_ALPHA8, glow::UNSIGNED_BYTE))
        }
        (Format::SRGBA(Size::Eight, Size::Eight, Size::Eight, Size::Eight), Type::NormIntegral) => {
            Some((glow::RGBA, glow::SRGB8_ALPHA8, glow::BYTE))
        }

        (Format::Depth(Size::ThirtyTwo), Type::Floating) => {
            Some((glow::DEPTH_COMPONENT, glow::DEPTH_COMPONENT32F, glow::FLOAT))
        }

        _ => None,
    }
}
