use luminance::backend::texture::{Texture as TextureBackend, TextureBase};
use luminance::depth_test::DepthComparison;
use luminance::pixel::{Pixel, PixelFormat};
use luminance::texture::{
  Dim, Dimensionable, GenMipmaps, MagFilter, MinFilter, Sampler, TextureError, Wrap,
};
use std::cell::RefCell;
use std::mem;
use std::rc::Rc;
use std::slice;
use web_sys::{WebGl2RenderingContext, WebGlTexture};

use crate::glow::array_buffer::IntoArrayBuffer;
use crate::glow::pixel::webgl_pixel_format;
use crate::glow::state::WebGL2State;
use crate::glow::WebGL2;

pub struct Texture {
  pub(crate) handle: WebGlTexture,
  pub(crate) target: u32, // “type” of the texture; used for bindings
  mipmaps: usize,
  state: Rc<RefCell<WebGL2State>>,
}

impl Texture {
  pub(crate) fn handle(&self) -> &WebGlTexture {
    &self.handle
  }
}

impl Drop for Texture {
  fn drop(&mut self) {
    self
      .state
      .borrow_mut()
      .ctx
      .delete_texture(Some(&self.handle));
  }
}

unsafe impl TextureBase for WebGL2 {
  type TextureRepr = Texture;
}

unsafe impl<D, P> TextureBackend<D, P> for WebGL2
where
  D: Dimensionable,
  P: Pixel,
  P::Encoding: IntoArrayBuffer,
  P::RawEncoding: IntoArrayBuffer,
{
  unsafe fn new_texture(
    &mut self,
    size: D::Size,
    mipmaps: usize,
    sampler: Sampler,
  ) -> Result<Self::TextureRepr, TextureError> {
    let mipmaps = mipmaps + 1; // + 1 prevent having 0 mipmaps
    let dim = D::dim();
    let target = opengl_target(dim).ok_or_else(|| {
      TextureError::TextureStorageCreationFailed(format!("incompatible texture dim: {}", dim))
    })?;

    let mut state = self.state.borrow_mut();

    let handle = state.create_texture().ok_or_else(|| {
      TextureError::TextureStorageCreationFailed("cannot create texture".to_owned())
    })?;
    state.bind_texture(target, Some(&handle));

    setup_texture::<D>(
      &mut state,
      target,
      size,
      mipmaps,
      P::pixel_format(),
      sampler,
    )?;

    let texture = Texture {
      handle,
      target,
      mipmaps,
      state: self.state.clone(),
    };

    Ok(texture)
  }

  unsafe fn mipmaps(texture: &Self::TextureRepr) -> usize {
    texture.mipmaps
  }

  unsafe fn clear_part(
    texture: &mut Self::TextureRepr,
    gen_mipmaps: GenMipmaps,
    offset: D::Offset,
    size: D::Size,
    pixel: P::Encoding,
  ) -> Result<(), TextureError> {
    <Self as TextureBackend<D, P>>::upload_part(
      texture,
      gen_mipmaps,
      offset,
      size,
      &vec![pixel; dim_capacity::<D>(size) as usize],
    )
  }

  unsafe fn clear(
    texture: &mut Self::TextureRepr,
    gen_mipmaps: GenMipmaps,
    size: D::Size,
    pixel: P::Encoding,
  ) -> Result<(), TextureError> {
    <Self as TextureBackend<D, P>>::clear_part(texture, gen_mipmaps, D::ZERO_OFFSET, size, pixel)
  }

  unsafe fn upload_part(
    texture: &mut Self::TextureRepr,
    gen_mipmaps: GenMipmaps,
    offset: D::Offset,
    size: D::Size,
    texels: &[P::Encoding],
  ) -> Result<(), TextureError> {
    let mut gfx_state = texture.state.borrow_mut();

    gfx_state.bind_texture(texture.target, Some(&texture.handle));

    upload_texels::<D, P, P::Encoding>(&mut gfx_state, texture.target, offset, size, texels)?;

    if gen_mipmaps == GenMipmaps::Yes {
      gfx_state.ctx.generate_mipmap(texture.target);
    }

    Ok(())
  }

  unsafe fn upload(
    texture: &mut Self::TextureRepr,
    gen_mipmaps: GenMipmaps,
    size: D::Size,
    texels: &[P::Encoding],
  ) -> Result<(), TextureError> {
    <Self as TextureBackend<D, P>>::upload_part(texture, gen_mipmaps, D::ZERO_OFFSET, size, texels)
  }

  unsafe fn upload_part_raw(
    texture: &mut Self::TextureRepr,
    gen_mipmaps: GenMipmaps,
    offset: D::Offset,
    size: D::Size,
    texels: &[P::RawEncoding],
  ) -> Result<(), TextureError> {
    let mut gfx_state = texture.state.borrow_mut();

    gfx_state.bind_texture(texture.target, Some(&texture.handle));

    upload_texels::<D, P, P::RawEncoding>(&mut gfx_state, texture.target, offset, size, texels)?;

    if gen_mipmaps == GenMipmaps::Yes {
      gfx_state.ctx.generate_mipmap(texture.target);
    }

    Ok(())
  }

  unsafe fn upload_raw(
    texture: &mut Self::TextureRepr,
    gen_mipmaps: GenMipmaps,
    size: D::Size,
    texels: &[P::RawEncoding],
  ) -> Result<(), TextureError> {
    <Self as TextureBackend<D, P>>::upload_part_raw(
      texture,
      gen_mipmaps,
      D::ZERO_OFFSET,
      size,
      texels,
    )
  }

  unsafe fn get_raw_texels(
    texture: &Self::TextureRepr,
    size: D::Size,
  ) -> Result<Vec<P::RawEncoding>, TextureError>
  where
    P::RawEncoding: Copy + Default,
  {
    let pf = P::pixel_format();
    let (format, _, ty) = webgl_pixel_format(pf).ok_or(TextureError::UnsupportedPixelFormat(pf))?;

    let mut gfx_state = texture.state.borrow_mut();
    gfx_state.bind_texture(texture.target, Some(&texture.handle));

    // Retrieve the size of the texture (w and h); WebGL2 doesn’t support the
    // glGetTexLevelParameteriv function (I know it’s fucking surprising), so we have to implement
    // a workaround and store that value on the CPU side.
    let w = D::width(size);
    let h = D::height(size);

    // set the packing alignment based on the number of bytes to skip
    let skip_bytes = (pf.format.size() * w as usize) % 8;
    set_pack_alignment(&mut gfx_state, skip_bytes);

    // We need a workaround to get the texel data, because WebGL2 doesn’t support the glGetTexImage
    // function. The idea is that we are using a special read framebuffer that is always around and
    // on which we can attach the texture we want to read the texels from.
    match gfx_state.create_or_get_readback_framebuffer() {
      Some(ref readback_fb) => {
        // Resize the vec to allocate enough space to host the returned texels.
        let texels_nb = (w * h) as usize * pf.canals_len();
        let mut texels = vec![Default::default(); texels_nb];

        // Attach the texture so that we can read from the framebuffer; careful here, since we are
        // reading from a 2D texture while the attached texture might not be compatible.
        gfx_state.bind_read_framebuffer(Some(readback_fb));
        gfx_state.ctx.framebuffer_texture_2d(
          WebGl2RenderingContext::READ_FRAMEBUFFER,
          WebGl2RenderingContext::COLOR_ATTACHMENT0,
          texture.target,
          Some(&texture.handle),
          0,
        );

        // Read from the framebuffer.
        gfx_state
          .ctx
          .read_pixels_with_u8_array_and_dst_offset(
            0,
            0,
            w as i32,
            h as i32,
            format,
            ty,
            slice::from_raw_parts_mut(
              texels.as_mut_ptr() as *mut u8,
              texels_nb * mem::size_of::<P::RawEncoding>(),
            ),
            0,
          )
          .map_err(|e| TextureError::CannotRetrieveTexels(format!("{:?}", e)))?;

        // Detach the texture from the framebuffer.
        gfx_state.ctx.framebuffer_texture_2d(
          WebGl2RenderingContext::READ_FRAMEBUFFER,
          WebGl2RenderingContext::COLOR_ATTACHMENT0,
          texture.target,
          None,
          0,
        );

        Ok(texels)
      }

      None => Err(TextureError::cannot_retrieve_texels(
        "unavailable readback framebuffer",
      )),
    }
  }
}

pub(crate) fn opengl_target(d: Dim) -> Option<u32> {
  match d {
    Dim::Dim2 => Some(WebGl2RenderingContext::TEXTURE_2D),
    Dim::Dim3 => Some(WebGl2RenderingContext::TEXTURE_3D),
    Dim::Cubemap => Some(WebGl2RenderingContext::TEXTURE_CUBE_MAP),
    Dim::Dim2Array => Some(WebGl2RenderingContext::TEXTURE_2D_ARRAY),
    _ => None,
  }
}

/// Set all the required internal state required for the texture to be valid.
pub(crate) unsafe fn setup_texture<D>(
  state: &mut WebGL2State,
  target: u32,
  size: D::Size,
  mipmaps: usize,
  pf: PixelFormat,
  sampler: Sampler,
) -> Result<(), TextureError>
where
  D: Dimensionable,
{
  set_texture_levels(state, target, mipmaps);
  apply_sampler_to_texture(state, target, sampler);
  create_texture_storage::<D>(state, size, mipmaps, pf)
}

fn set_texture_levels(state: &mut WebGL2State, target: u32, mipmaps: usize) {
  state
    .ctx
    .tex_parameteri(target, WebGl2RenderingContext::TEXTURE_BASE_LEVEL, 0);

  state.ctx.tex_parameteri(
    target,
    WebGl2RenderingContext::TEXTURE_MAX_LEVEL,
    mipmaps as i32 - 1,
  );
}

fn apply_sampler_to_texture(state: &mut WebGL2State, target: u32, sampler: Sampler) {
  state.ctx.tex_parameteri(
    target,
    WebGl2RenderingContext::TEXTURE_WRAP_R,
    webgl_wrap(sampler.wrap_r) as i32,
  );
  state.ctx.tex_parameteri(
    target,
    WebGl2RenderingContext::TEXTURE_WRAP_S,
    webgl_wrap(sampler.wrap_s) as i32,
  );
  state.ctx.tex_parameteri(
    target,
    WebGl2RenderingContext::TEXTURE_WRAP_T,
    webgl_wrap(sampler.wrap_t) as i32,
  );
  state.ctx.tex_parameteri(
    target,
    WebGl2RenderingContext::TEXTURE_MIN_FILTER,
    webgl_min_filter(sampler.min_filter) as i32,
  );
  state.ctx.tex_parameteri(
    target,
    WebGl2RenderingContext::TEXTURE_MAG_FILTER,
    webgl_mag_filter(sampler.mag_filter) as i32,
  );

  match sampler.depth_comparison {
    Some(fun) => {
      state.ctx.tex_parameteri(
        target,
        WebGl2RenderingContext::TEXTURE_COMPARE_FUNC,
        webgl_depth_comparison(fun) as i32,
      );
      state.ctx.tex_parameteri(
        target,
        WebGl2RenderingContext::TEXTURE_COMPARE_MODE,
        WebGl2RenderingContext::COMPARE_REF_TO_TEXTURE as i32,
      );
    }

    None => {
      state.ctx.tex_parameteri(
        target,
        WebGl2RenderingContext::TEXTURE_COMPARE_MODE,
        WebGl2RenderingContext::NONE as i32,
      );
    }
  }
}

fn webgl_wrap(wrap: Wrap) -> u32 {
  match wrap {
    Wrap::ClampToEdge => WebGl2RenderingContext::CLAMP_TO_EDGE,
    Wrap::Repeat => WebGl2RenderingContext::REPEAT,
    Wrap::MirroredRepeat => WebGl2RenderingContext::MIRRORED_REPEAT,
  }
}

fn webgl_min_filter(filter: MinFilter) -> u32 {
  match filter {
    MinFilter::Nearest => WebGl2RenderingContext::NEAREST,
    MinFilter::Linear => WebGl2RenderingContext::LINEAR,
    MinFilter::NearestMipmapNearest => WebGl2RenderingContext::NEAREST_MIPMAP_NEAREST,
    MinFilter::NearestMipmapLinear => WebGl2RenderingContext::NEAREST_MIPMAP_LINEAR,
    MinFilter::LinearMipmapNearest => WebGl2RenderingContext::LINEAR_MIPMAP_NEAREST,
    MinFilter::LinearMipmapLinear => WebGl2RenderingContext::LINEAR_MIPMAP_LINEAR,
  }
}

fn webgl_mag_filter(filter: MagFilter) -> u32 {
  match filter {
    MagFilter::Nearest => WebGl2RenderingContext::NEAREST,
    MagFilter::Linear => WebGl2RenderingContext::LINEAR,
  }
}

fn create_texture_storage<D>(
  state: &mut WebGL2State,
  size: D::Size,
  mipmaps: usize,
  pf: PixelFormat,
) -> Result<(), TextureError>
where
  D: Dimensionable,
{
  match webgl_pixel_format(pf) {
    Some(glf) => {
      let (format, iformat, encoding) = glf;

      match D::dim() {
        // 2D texture
        Dim::Dim2 => {
          create_texture_2d_storage(
            state,
            WebGl2RenderingContext::TEXTURE_2D,
            iformat,
            D::width(size),
            D::height(size),
            mipmaps,
          )?;
          Ok(())
        }

        // 3D texture
        Dim::Dim3 => {
          create_texture_3d_storage(
            state,
            WebGl2RenderingContext::TEXTURE_3D,
            iformat,
            D::width(size),
            D::height(size),
            D::depth(size),
            mipmaps,
          )?;
          Ok(())
        }

        // cubemap
        Dim::Cubemap => {
          create_cubemap_storage(state, format, iformat, encoding, D::width(size), mipmaps)?;
          Ok(())
        }

        // 2D array texture
        Dim::Dim2Array => {
          create_texture_3d_storage(
            state,
            WebGl2RenderingContext::TEXTURE_2D_ARRAY,
            iformat,
            D::width(size),
            D::height(size),
            D::depth(size),
            mipmaps,
          )?;
          Ok(())
        }

        _ => Err(TextureError::unsupported_pixel_format(pf)),
      }
    }

    None => Err(TextureError::unsupported_pixel_format(pf)),
  }
}

fn create_texture_2d_storage(
  state: &mut WebGL2State,
  target: u32,
  iformat: u32,
  w: u32,
  h: u32,
  mipmaps: usize,
) -> Result<(), TextureError> {
  state
    .ctx
    .tex_storage_2d(target, mipmaps as i32, iformat, w as i32, h as i32);

  Ok(())
}

fn create_texture_3d_storage(
  state: &mut WebGL2State,
  target: u32,
  iformat: u32,
  w: u32,
  h: u32,
  d: u32,
  mipmaps: usize,
) -> Result<(), TextureError> {
  state.ctx.tex_storage_3d(
    target,
    mipmaps as i32,
    iformat,
    w as i32,
    h as i32,
    d as i32,
  );

  Ok(())
}

fn create_cubemap_storage(
  state: &mut WebGL2State,
  format: u32,
  iformat: u32,
  encoding: u32,
  s: u32,
  mipmaps: usize,
) -> Result<(), TextureError> {
  for level in 0..mipmaps {
    let s = s / (1 << level as u32);

    for face in 0..6 {
      state
        .ctx
        .tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_i32(
          WebGl2RenderingContext::TEXTURE_CUBE_MAP_POSITIVE_X + face,
          level as i32,
          iformat as i32,
          s as i32,
          s as i32,
          0,
          format,
          encoding,
          0,
        )
        .map_err(|e| TextureError::TextureStorageCreationFailed(format!("{:?}", e)))?;
    }
  }

  Ok(())
}

// set the unpack alignment for uploading aligned texels
fn set_unpack_alignment(state: &mut WebGL2State, skip_bytes: usize) {
  let unpack_alignment = match skip_bytes {
    0 => 8,
    2 => 2,
    4 => 4,
    _ => 1,
  } as i32;

  state
    .ctx
    .pixel_storei(WebGl2RenderingContext::UNPACK_ALIGNMENT, unpack_alignment);
}

// set the pack alignment for downloading aligned texels
fn set_pack_alignment(state: &mut WebGL2State, skip_bytes: usize) {
  let pack_alignment = match skip_bytes {
    0 => 8,
    2 => 2,
    4 => 4,
    _ => 1,
  } as i32;

  state
    .ctx
    .pixel_storei(WebGl2RenderingContext::PACK_ALIGNMENT, pack_alignment);
}
// Upload texels into the texture’s memory. Becareful of the type of texels you send down.
fn upload_texels<D, P, T>(
  state: &mut WebGL2State,
  target: u32,
  off: D::Offset,
  size: D::Size,
  texels: &[T],
) -> Result<(), TextureError>
where
  D: Dimensionable,
  P: Pixel,
  T: IntoArrayBuffer,
{
  // number of bytes in the input texels argument
  let input_bytes = texels.len() * mem::size_of::<T>();
  let pf = P::pixel_format();
  let pf_size = pf.format.size();
  let expected_bytes = D::count(size) * pf_size;

  if input_bytes < expected_bytes {
    // potential segfault / overflow; abort
    return Err(TextureError::not_enough_pixels(expected_bytes, input_bytes));
  }

  // set the pixel row alignment to the required value for uploading data according to the width
  // of the texture and the size of a single pixel; here, skip_bytes represents the number of bytes
  // that will be skipped
  let skip_bytes = (D::width(size) as usize * pf_size) % 8;
  set_unpack_alignment(state, skip_bytes);

  // coerce the texels slice into a web-sys array buffer so that we can pass them to the super ugly
  // method below
  let array_buffer;
  unsafe {
    array_buffer = T::into_array_buffer(texels);
  }

  match webgl_pixel_format(pf) {
    Some((format, _, encoding)) => match D::dim() {
      Dim::Dim2 => {
        state
          .ctx
          .tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_array_buffer_view_and_src_offset(
            target,
            0,
            D::x_offset(off) as i32,
            D::y_offset(off) as i32,
            D::width(size) as i32,
            D::height(size) as i32,
            format,
            encoding,
            &array_buffer,
            0,
          )
          .map_err(|e| TextureError::CannotUploadTexels(format!("{:?}", e)))?;
      }

      Dim::Dim3 => {
        state
          .ctx
          .tex_sub_image_3d_with_opt_array_buffer_view(
            target,
            0,
            D::x_offset(off) as i32,
            D::y_offset(off) as i32,
            D::z_offset(off) as i32,
            D::width(size) as i32,
            D::height(size) as i32,
            D::depth(size) as i32,
            format,
            encoding,
            Some(&array_buffer),
          )
          .map_err(|e| TextureError::CannotUploadTexels(format!("{:?}", e)))?;
      }

      Dim::Cubemap => {
        state
          .ctx
          .tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_array_buffer_view_and_src_offset(
            WebGl2RenderingContext::TEXTURE_CUBE_MAP_POSITIVE_X + D::z_offset(off),
            0,
            D::x_offset(off) as i32,
            D::y_offset(off) as i32,
            D::width(size) as i32,
            D::height(size) as i32,
            format,
            encoding,
            &array_buffer,
            0,
          )
          .map_err(|e| TextureError::CannotUploadTexels(format!("{:?}", e)))?;
      }

      Dim::Dim2Array => {
        state
          .ctx
          .tex_sub_image_3d_with_opt_array_buffer_view(
            target,
            0,
            D::x_offset(off) as i32,
            D::y_offset(off) as i32,
            D::z_offset(off) as i32,
            D::width(size) as i32,
            D::height(size) as i32,
            D::depth(size) as i32,
            format,
            encoding,
            Some(&array_buffer),
          )
          .map_err(|e| TextureError::CannotUploadTexels(format!("{:?}", e)))?;
      }

      _ => return Err(TextureError::unsupported_pixel_format(pf)),
    },

    None => return Err(TextureError::unsupported_pixel_format(pf)),
  }

  Ok(())
}

// Capacity of the dimension, which is the product of the width, height and depth.
fn dim_capacity<D>(size: D::Size) -> u32
where
  D: Dimensionable,
{
  D::width(size) * D::height(size) * D::depth(size)
}

pub(crate) fn webgl_depth_comparison(dc: DepthComparison) -> u32 {
  match dc {
    DepthComparison::Never => WebGl2RenderingContext::NEVER,
    DepthComparison::Always => WebGl2RenderingContext::ALWAYS,
    DepthComparison::Equal => WebGl2RenderingContext::EQUAL,
    DepthComparison::NotEqual => WebGl2RenderingContext::NOTEQUAL,
    DepthComparison::Less => WebGl2RenderingContext::LESS,
    DepthComparison::LessOrEqual => WebGl2RenderingContext::LEQUAL,
    DepthComparison::Greater => WebGl2RenderingContext::GREATER,
    DepthComparison::GreaterOrEqual => WebGl2RenderingContext::GEQUAL,
  }
}
