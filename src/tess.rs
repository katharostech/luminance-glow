//! GlowBackend tessellation implementation.

use luminance::backend::buffer::BufferSlice as _;
use luminance::backend::tess::{
    IndexSlice as IndexSliceBackend, InstanceSlice as InstanceSliceBackend, Tess as TessBackend,
    VertexSlice as VertexSliceBackend,
};
use luminance::tess::{
    Deinterleaved, DeinterleavedData, Interleaved, Mode, TessError, TessIndex, TessIndexType,
    TessMapError, TessVertexData,
};
use luminance::vertex::{
    Deinterleave, Normalized, Vertex, VertexAttribDesc, VertexAttribDim, VertexAttribType,
    VertexBufferDesc, VertexInstancing,
};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::buffer::{Buffer, BufferSlice, BufferSliceMut};
use crate::state::{Bind, GlowState};
use crate::GlowBackend;
use glow::HasContext;

#[derive(Debug)]
struct TessRaw<I>
where
    I: TessIndex,
{
    vao: glow::VertexArray,
    mode: u32,
    vert_nb: usize,
    inst_nb: usize,
    // A small note: GlowBackend doesn’t support custom primitive restart index; it assumes the maximum
    // value of I as being that restart index.
    index_buffer: Option<Buffer<I>>,
    state: Rc<RefCell<GlowState>>,
}

impl<I> TessRaw<I>
where
    I: TessIndex,
{
    unsafe fn render(
        &self,
        start_index: usize,
        vert_nb: usize,
        inst_nb: usize,
    ) -> Result<(), TessError> {
        let vert_nb = vert_nb as _;
        let inst_nb = inst_nb as _;

        let mut gfx_st = self.state.borrow_mut();
        gfx_st.bind_vertex_array(Some(&self.vao), Bind::Cached);

        match (I::INDEX_TYPE, self.index_buffer.as_ref()) {
            (Some(index_ty), Some(_)) => {
                // indexed render
                let first = (index_ty.bytes() * start_index) as _;

                if inst_nb <= 1 {
                    gfx_st.ctx.draw_elements(
                        self.mode,
                        vert_nb,
                        index_type_to_glenum(index_ty),
                        first,
                    );
                } else {
                    gfx_st.ctx.draw_elements_instanced(
                        self.mode,
                        vert_nb,
                        index_type_to_glenum(index_ty),
                        first,
                        inst_nb,
                    );
                }
            }

            _ => {
                // direct render
                let first = start_index as _;

                if inst_nb <= 1 {
                    gfx_st.ctx.draw_arrays(self.mode, first, vert_nb);
                } else {
                    gfx_st
                        .ctx
                        .draw_arrays_instanced(self.mode, first, vert_nb, inst_nb);
                }
            }
        }

        Ok(())
    }
}

impl<I> Drop for TessRaw<I>
where
    I: TessIndex,
{
    fn drop(&mut self) {
        let mut state = self.state.borrow_mut();
        state.bind_vertex_array(None, Bind::Cached);
        unsafe {
            state.ctx.delete_vertex_array(self.vao);
        }
    }
}

#[derive(Debug)]
pub struct InterleavedTess<V, I, W>
where
    V: Vertex,
    I: TessIndex,
    W: Vertex,
{
    raw: TessRaw<I>,
    vertex_buffer: Option<Buffer<V>>,
    instance_buffer: Option<Buffer<W>>,
}

unsafe impl<V, I, W> TessBackend<V, I, W, Interleaved> for GlowBackend
where
    V: TessVertexData<Interleaved, Data = Vec<V>>,
    I: TessIndex,
    W: TessVertexData<Interleaved, Data = Vec<W>>,
{
    type TessRepr = InterleavedTess<V, I, W>;

    unsafe fn build(
        &mut self,
        vertex_data: Option<V::Data>,
        index_data: Vec<I>,
        instance_data: Option<W::Data>,
        mode: Mode,
        vert_nb: usize,
        inst_nb: usize,
        _: Option<I>,
    ) -> Result<Self::TessRepr, TessError> {
        let vao = self.state.borrow_mut().create_vertex_array().map_err(|e| {
            TessError::cannot_create(format!("the backend failed to create the VAO: {}", e))
        })?;

        // force binding the vertex array so that previously bound vertex arrays (possibly the same
        // handle) don’t prevent us from binding here
        self.state
            .borrow_mut()
            .bind_vertex_array(Some(&vao), Bind::Forced);

        let vertex_buffer = build_interleaved_vertex_buffer(self, vertex_data)?;
        let index_buffer = build_index_buffer(self, index_data)?;
        let instance_buffer = build_interleaved_vertex_buffer(self, instance_data)?;

        let mode = glow_mode(mode).ok_or_else(|| TessError::ForbiddenPrimitiveMode(mode))?;
        let state = self.state.clone();
        let raw = TessRaw {
            vao,
            mode,
            vert_nb,
            inst_nb,
            index_buffer,
            state,
        };

        Ok(InterleavedTess {
            raw,
            vertex_buffer,
            instance_buffer,
        })
    }

    unsafe fn tess_vertices_nb(tess: &Self::TessRepr) -> usize {
        tess.raw.vert_nb
    }

    unsafe fn tess_instances_nb(tess: &Self::TessRepr) -> usize {
        tess.raw.inst_nb
    }

    unsafe fn render(
        tess: &Self::TessRepr,
        start_index: usize,
        vert_nb: usize,
        inst_nb: usize,
    ) -> Result<(), TessError> {
        tess.raw.render(start_index, vert_nb, inst_nb)
    }
}

unsafe impl<V, I, W> VertexSliceBackend<V, I, W, Interleaved, V> for GlowBackend
where
    V: TessVertexData<Interleaved, Data = Vec<V>>,
    I: TessIndex,
    W: TessVertexData<Interleaved, Data = Vec<W>>,
{
    type VertexSliceRepr = BufferSlice<V>;
    type VertexSliceMutRepr = BufferSliceMut<V>;

    unsafe fn vertices(tess: &mut Self::TessRepr) -> Result<Self::VertexSliceRepr, TessMapError> {
        match tess.vertex_buffer {
            Some(ref vb) => Ok(GlowBackend::slice_buffer(vb)?),
            None => Err(TessMapError::forbidden_attributeless_mapping()),
        }
    }

    unsafe fn vertices_mut(
        tess: &mut Self::TessRepr,
    ) -> Result<Self::VertexSliceMutRepr, TessMapError> {
        match tess.vertex_buffer {
            Some(ref mut vb) => Ok(GlowBackend::slice_buffer_mut(vb)?),
            None => Err(TessMapError::forbidden_attributeless_mapping()),
        }
    }
}

unsafe impl<V, I, W> IndexSliceBackend<V, I, W, Interleaved> for GlowBackend
where
    V: TessVertexData<Interleaved, Data = Vec<V>>,
    I: TessIndex,
    W: TessVertexData<Interleaved, Data = Vec<W>>,
{
    type IndexSliceRepr = BufferSlice<I>;
    type IndexSliceMutRepr = BufferSliceMut<I>;

    unsafe fn indices(tess: &mut Self::TessRepr) -> Result<Self::IndexSliceRepr, TessMapError> {
        match tess.raw.index_buffer {
            Some(ref buffer) => Ok(GlowBackend::slice_buffer(buffer)?),
            None => Err(TessMapError::forbidden_attributeless_mapping()),
        }
    }

    unsafe fn indices_mut(
        tess: &mut Self::TessRepr,
    ) -> Result<Self::IndexSliceMutRepr, TessMapError> {
        match tess.raw.index_buffer {
            Some(ref mut buffer) => Ok(GlowBackend::slice_buffer_mut(buffer)?),
            None => Err(TessMapError::forbidden_attributeless_mapping()),
        }
    }
}

unsafe impl<V, I, W> InstanceSliceBackend<V, I, W, Interleaved, W> for GlowBackend
where
    V: TessVertexData<Interleaved, Data = Vec<V>>,
    I: TessIndex,
    W: TessVertexData<Interleaved, Data = Vec<W>>,
{
    type InstanceSliceRepr = BufferSlice<W>;
    type InstanceSliceMutRepr = BufferSliceMut<W>;

    unsafe fn instances(
        tess: &mut Self::TessRepr,
    ) -> Result<Self::InstanceSliceRepr, TessMapError> {
        match tess.instance_buffer {
            Some(ref vb) => Ok(GlowBackend::slice_buffer(vb)?),
            None => Err(TessMapError::forbidden_attributeless_mapping()),
        }
    }

    unsafe fn instances_mut(
        tess: &mut Self::TessRepr,
    ) -> Result<Self::InstanceSliceMutRepr, TessMapError> {
        match tess.instance_buffer {
            Some(ref mut vb) => Ok(GlowBackend::slice_buffer_mut(vb)?),
            None => Err(TessMapError::forbidden_attributeless_mapping()),
        }
    }
}

#[derive(Debug)]
pub struct DeinterleavedTess<V, I, W>
where
    V: Vertex,
    I: TessIndex,
    W: Vertex,
{
    raw: TessRaw<I>,
    vertex_buffers: Vec<Buffer<u8>>,
    instance_buffers: Vec<Buffer<u8>>,
    _phantom: PhantomData<*const (V, W)>,
}

unsafe impl<V, I, W> TessBackend<V, I, W, Deinterleaved> for GlowBackend
where
    V: TessVertexData<Deinterleaved, Data = Vec<DeinterleavedData>>,
    I: TessIndex,
    W: TessVertexData<Deinterleaved, Data = Vec<DeinterleavedData>>,
{
    type TessRepr = DeinterleavedTess<V, I, W>;

    unsafe fn build(
        &mut self,
        vertex_data: Option<V::Data>,
        index_data: Vec<I>,
        instance_data: Option<W::Data>,
        mode: Mode,
        vert_nb: usize,
        inst_nb: usize,
        _: Option<I>,
    ) -> Result<Self::TessRepr, TessError> {
        let vao = self.state.borrow_mut().create_vertex_array().map_err(|e| {
            TessError::cannot_create(format!("the backend failed to create the VAO: {}", e))
        })?;

        // force binding the vertex array so that previously bound vertex arrays (possibly the same
        // handle) don’t prevent us from binding here
        self.state
            .borrow_mut()
            .bind_vertex_array(Some(&vao), Bind::Forced);

        let vertex_buffers = build_deinterleaved_vertex_buffers::<V>(self, vertex_data)?;
        let index_buffer = build_index_buffer(self, index_data)?;
        let instance_buffers = build_deinterleaved_vertex_buffers::<W>(self, instance_data)?;

        let mode = glow_mode(mode).ok_or_else(|| TessError::ForbiddenPrimitiveMode(mode))?;
        let state = self.state.clone();
        let raw = TessRaw {
            vao,
            mode,
            vert_nb,
            inst_nb,
            index_buffer,
            state,
        };

        Ok(DeinterleavedTess {
            raw,
            vertex_buffers,
            instance_buffers,
            _phantom: PhantomData,
        })
    }

    unsafe fn tess_vertices_nb(tess: &Self::TessRepr) -> usize {
        tess.raw.vert_nb
    }

    unsafe fn tess_instances_nb(tess: &Self::TessRepr) -> usize {
        tess.raw.inst_nb
    }

    unsafe fn render(
        tess: &Self::TessRepr,
        start_index: usize,
        vert_nb: usize,
        inst_nb: usize,
    ) -> Result<(), TessError> {
        tess.raw.render(start_index, vert_nb, inst_nb)
    }
}

unsafe impl<V, I, W, T> VertexSliceBackend<V, I, W, Deinterleaved, T> for GlowBackend
where
    V: TessVertexData<Deinterleaved, Data = Vec<DeinterleavedData>> + Deinterleave<T>,
    I: TessIndex,
    W: TessVertexData<Deinterleaved, Data = Vec<DeinterleavedData>>,
{
    type VertexSliceRepr = BufferSlice<T>;
    type VertexSliceMutRepr = BufferSliceMut<T>;

    unsafe fn vertices(tess: &mut Self::TessRepr) -> Result<Self::VertexSliceRepr, TessMapError> {
        if tess.vertex_buffers.is_empty() {
            Err(TessMapError::forbidden_attributeless_mapping())
        } else {
            let buffer = &tess.vertex_buffers[V::RANK];
            let slice = GlowBackend::slice_buffer(buffer)?.transmute();
            Ok(slice)
        }
    }

    unsafe fn vertices_mut(
        tess: &mut Self::TessRepr,
    ) -> Result<Self::VertexSliceMutRepr, TessMapError> {
        if tess.vertex_buffers.is_empty() {
            Err(TessMapError::forbidden_attributeless_mapping())
        } else {
            let buffer = &mut tess.vertex_buffers[V::RANK];
            let slice = GlowBackend::slice_buffer_mut(buffer)?.transmute();
            Ok(slice)
        }
    }
}

unsafe impl<V, I, W> IndexSliceBackend<V, I, W, Deinterleaved> for GlowBackend
where
    V: TessVertexData<Deinterleaved, Data = Vec<DeinterleavedData>>,
    I: TessIndex,
    W: TessVertexData<Deinterleaved, Data = Vec<DeinterleavedData>>,
{
    type IndexSliceRepr = BufferSlice<I>;
    type IndexSliceMutRepr = BufferSliceMut<I>;

    unsafe fn indices(tess: &mut Self::TessRepr) -> Result<Self::IndexSliceRepr, TessMapError> {
        match tess.raw.index_buffer {
            Some(ref buffer) => Ok(GlowBackend::slice_buffer(buffer)?),
            None => Err(TessMapError::forbidden_attributeless_mapping()),
        }
    }

    unsafe fn indices_mut(
        tess: &mut Self::TessRepr,
    ) -> Result<Self::IndexSliceMutRepr, TessMapError> {
        match tess.raw.index_buffer {
            Some(ref mut buffer) => Ok(GlowBackend::slice_buffer_mut(buffer)?),
            None => Err(TessMapError::forbidden_attributeless_mapping()),
        }
    }
}

unsafe impl<V, I, W, T> InstanceSliceBackend<V, I, W, Deinterleaved, T> for GlowBackend
where
    V: TessVertexData<Deinterleaved, Data = Vec<DeinterleavedData>>,
    I: TessIndex,
    W: TessVertexData<Deinterleaved, Data = Vec<DeinterleavedData>> + Deinterleave<T>,
{
    type InstanceSliceRepr = BufferSlice<T>;
    type InstanceSliceMutRepr = BufferSliceMut<T>;

    unsafe fn instances(
        tess: &mut Self::TessRepr,
    ) -> Result<Self::InstanceSliceRepr, TessMapError> {
        if tess.instance_buffers.is_empty() {
            Err(TessMapError::forbidden_attributeless_mapping())
        } else {
            let buffer = &tess.instance_buffers[W::RANK];
            let slice = GlowBackend::slice_buffer(buffer)?.transmute();
            Ok(slice)
        }
    }

    unsafe fn instances_mut(
        tess: &mut Self::TessRepr,
    ) -> Result<Self::InstanceSliceMutRepr, TessMapError> {
        if tess.instance_buffers.is_empty() {
            Err(TessMapError::forbidden_attributeless_mapping())
        } else {
            let buffer = &mut tess.instance_buffers[W::RANK];
            let slice = GlowBackend::slice_buffer_mut(buffer)?.transmute();
            Ok(slice)
        }
    }
}

fn build_interleaved_vertex_buffer<V>(
    backend: &mut GlowBackend,
    vertices: Option<Vec<V>>,
) -> Result<Option<Buffer<V>>, TessError>
where
    V: Vertex,
{
    match vertices {
        Some(vertices) => {
            let fmt = V::vertex_desc();

            let vb = if vertices.is_empty() {
                None
            } else {
                let vb = Buffer::from_vec(backend, vertices, glow::ARRAY_BUFFER)?;

                // force binding as it’s meaningful when a vao is bound
                backend
                    .state
                    .borrow_mut()
                    .bind_array_buffer(Some(vb.handle()), Bind::Forced);
                set_vertex_pointers(&mut backend.state.borrow_mut().ctx, &fmt);

                Some(vb)
            };

            Ok(vb)
        }

        None => Ok(None),
    }
}

fn build_deinterleaved_vertex_buffers<V>(
    backend: &mut GlowBackend,
    vertices: Option<Vec<DeinterleavedData>>,
) -> Result<Vec<Buffer<u8>>, TessError>
where
    V: Vertex,
{
    match vertices {
        Some(attributes) => {
            attributes
                .into_iter()
                .zip(V::vertex_desc())
                .map(|(attribute, fmt)| {
                    let vb = Buffer::from_vec(backend, attribute.into_vec(), glow::ARRAY_BUFFER)?;

                    // force binding as it’s meaningful when a vao is bound
                    backend
                        .state
                        .borrow_mut()
                        .bind_array_buffer(Some(vb.handle()), Bind::Forced);
                    set_vertex_pointers(&mut backend.state.borrow_mut().ctx, &[fmt]);

                    Ok(vb)
                })
                .collect::<Result<Vec<_>, _>>()
        }

        None => Ok(Vec::new()),
    }
}

/// Turn a [`Vec`] of indices to a [`Buffer`], if indices are present.
fn build_index_buffer<I>(
    backend: &mut GlowBackend,
    data: Vec<I>,
) -> Result<Option<Buffer<I>>, TessError>
where
    I: TessIndex,
{
    let ib = if !data.is_empty() {
        let ib = Buffer::from_vec(backend, data, glow::ELEMENT_ARRAY_BUFFER)?;

        // force binding as it’s meaningful when a vao is bound
        backend
            .state
            .borrow_mut()
            .bind_element_array_buffer(Some(ib.handle()), Bind::Forced);

        Some(ib)
    } else {
        None
    };

    Ok(ib)
}

/// Give WebGL types information on the content of the VBO by setting vertex descriptors and
/// pointers to buffer memory.
fn set_vertex_pointers(ctx: &mut glow::Context, descriptors: &[VertexBufferDesc]) {
    // this function sets the vertex attribute pointer for the input list by computing:
    //   - The vertex attribute ID: this is the “rank” of the attribute in the input list (order
    //     matters, for short).
    //   - The stride: this is easily computed, since it’s the size (bytes) of a single vertex.
    //   - The offsets: each attribute has a given offset in the buffer. This is computed by
    //     accumulating the size of all previously set attributes.
    let offsets = aligned_offsets(descriptors);
    let vertex_weight = offset_based_vertex_weight(descriptors, &offsets);

    for (desc, off) in descriptors.iter().zip(offsets) {
        set_component_format(ctx, vertex_weight, off, desc);
    }
}

/// Compute offsets for all the vertex components according to the alignments provided.
fn aligned_offsets(descriptor: &[VertexBufferDesc]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(descriptor.len());
    let mut off = 0;

    // compute offsets
    for desc in descriptor {
        let desc = &desc.attrib_desc;
        off = off_align(off, desc.align); // keep the current component descriptor aligned
        offsets.push(off);
        off += component_weight(desc); // increment the offset by the pratical size of the component
    }

    offsets
}

/// Align an offset.
#[inline]
fn off_align(off: usize, align: usize) -> usize {
    let a = align - 1;
    (off + a) & !a
}

/// Weight in bytes of a vertex component.
fn component_weight(f: &VertexAttribDesc) -> usize {
    dim_as_size(f.dim) as usize * f.unit_size
}

fn dim_as_size(d: VertexAttribDim) -> usize {
    match d {
        VertexAttribDim::Dim1 => 1,
        VertexAttribDim::Dim2 => 2,
        VertexAttribDim::Dim3 => 3,
        VertexAttribDim::Dim4 => 4,
    }
}

/// Weight in bytes of a single vertex, taking into account padding so that the vertex stay correctly
/// aligned.
fn offset_based_vertex_weight(descriptors: &[VertexBufferDesc], offsets: &[usize]) -> usize {
    if descriptors.is_empty() || offsets.is_empty() {
        return 0;
    }

    off_align(
        offsets[offsets.len() - 1]
            + component_weight(&descriptors[descriptors.len() - 1].attrib_desc),
        descriptors[0].attrib_desc.align,
    )
}

/// Set the vertex component OpenGL pointers regarding the index of the component and the vertex
/// stride.
fn set_component_format(
    ctx: &mut glow::Context,
    stride: usize,
    off: usize,
    desc: &VertexBufferDesc,
) {
    unsafe {
        let attrib_desc = &desc.attrib_desc;
        let index = desc.index as u32;

        // FIXME: Not sure if I got this right
        match attrib_desc.ty {
            VertexAttribType::Floating => {
                ctx.vertex_attrib_pointer_f32(
                    index,
                    dim_as_size(attrib_desc.dim) as _,
                    glow_sized_type(&attrib_desc),
                    false,
                    stride as _,
                    off as _,
                );
            }

            VertexAttribType::Integral(Normalized::No)
            | VertexAttribType::Unsigned(Normalized::No)
            | VertexAttribType::Boolean => {
                // non-normalized integrals / booleans
                ctx.vertex_attrib_pointer_i32(
                    index,
                    dim_as_size(attrib_desc.dim) as _,
                    glow_sized_type(&attrib_desc),
                    stride as _,
                    off as _,
                );
            }

            _ => {
                // normalized integrals
                ctx.vertex_attrib_pointer_f32(
                    index,
                    dim_as_size(attrib_desc.dim) as _,
                    glow_sized_type(&attrib_desc),
                    true,
                    stride as _,
                    off as _,
                );
            }
        }

        // set vertex attribute divisor based on the vertex instancing configuration
        let divisor = match desc.instancing {
            VertexInstancing::On => 1,
            VertexInstancing::Off => 0,
        };

        ctx.vertex_attrib_divisor(index, divisor);

        ctx.enable_vertex_attrib_array(index);
    }
}

fn glow_sized_type(f: &VertexAttribDesc) -> u32 {
    match (f.ty, f.unit_size) {
        (VertexAttribType::Integral(_), 1) => glow::BYTE,
        (VertexAttribType::Integral(_), 2) => glow::SHORT,
        (VertexAttribType::Integral(_), 4) => glow::INT,
        (VertexAttribType::Unsigned(_), 1) | (VertexAttribType::Boolean, 1) => glow::UNSIGNED_BYTE,
        (VertexAttribType::Unsigned(_), 2) => glow::UNSIGNED_SHORT,
        (VertexAttribType::Unsigned(_), 4) => glow::UNSIGNED_INT,
        (VertexAttribType::Floating, 4) => glow::FLOAT,
        _ => panic!("unsupported vertex component format: {:?}", f),
    }
}

fn glow_mode(mode: Mode) -> Option<u32> {
    match mode {
        Mode::Point => Some(glow::POINTS),
        Mode::Line => Some(glow::LINES),
        Mode::LineStrip => Some(glow::LINE_STRIP),
        Mode::Triangle => Some(glow::TRIANGLES),
        Mode::TriangleFan => Some(glow::TRIANGLE_FAN),
        Mode::TriangleStrip => Some(glow::TRIANGLE_STRIP),
        Mode::Patch(_) => None,
    }
}

fn index_type_to_glenum(ty: TessIndexType) -> u32 {
    match ty {
        TessIndexType::U8 => glow::UNSIGNED_BYTE,
        TessIndexType::U16 => glow::UNSIGNED_SHORT,
        TessIndexType::U32 => glow::UNSIGNED_INT,
    }
}
