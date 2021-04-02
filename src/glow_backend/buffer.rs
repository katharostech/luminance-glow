//! WebGL2 buffer implementation.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::slice;

use crate::glow_backend::state::{Bind, GlowState};
use crate::glow_backend::GlowBackend;
use glow::HasContext;
use luminance::backend::buffer::{Buffer as BufferBackend, BufferSlice as BufferSliceBackend};
use luminance::buffer::BufferError;

/// Wrapped WebGL buffer.
///
/// Used to drop the buffer.
#[derive(Clone, Debug)]
struct BufferWrapper {
    handle: glow::Buffer,
    state: Rc<RefCell<GlowState>>,
}

impl Drop for BufferWrapper {
    fn drop(&mut self) {
        unsafe {
            let mut state = self.state.borrow_mut();

            state.unbind_buffer(&self.handle);
            state.ctx.delete_buffer(self.handle);
        }
    }
}

/// WebGL buffer.
#[derive(Clone, Debug)]
pub struct Buffer<T> {
    /// A cached version of the GPU buffer; emulate persistent mapping.
    pub(crate) buf: Vec<T>,
    gl_buf: BufferWrapper,
}

impl<T> Buffer<T> {
    /// Create a new buffer from a length and a type. This is needed to implement repeat without Default.
    ///
    /// The `target` parameter allows to create the buffer with
    /// [`glow::ARRAY_BUFFER`] or [`glow::ELEMENT_ARRAY_BUFFER`]
    /// directly, as WebGL2 doesn’t support changing the target type after the buffer is created.
    fn new(
        ctx: &mut GlowBackend,
        len: usize,
        clear_value: T,
        target: u32,
    ) -> Result<Self, BufferError>
    where
        T: Copy,
    {
        unsafe {
            let mut state = ctx.state.borrow_mut();

            let mut buf = Vec::new();
            buf.resize_with(len, || clear_value);

            // generate a buffer and force binding the handle; this prevent side-effects from previous bound
            // resources to prevent binding the buffer
            let handle = state
                .create_buffer()
                .map_err(|_| BufferError::cannot_create())?;

            Self::bind(&mut state, handle, target)?;

            let bytes = mem::size_of::<T>() * len;
            state
                .ctx
                .buffer_data_size(target, bytes as i32, glow::STREAM_DRAW);

            let gl_buf = BufferWrapper {
                handle,
                state: ctx.state.clone(),
            };

            Ok(Buffer { buf, gl_buf })
        }
    }

    pub(crate) fn from_vec(
        ctx: &mut GlowBackend,
        vec: Vec<T>,
        target: u32,
    ) -> Result<Self, BufferError> {
        unsafe {
            let mut state = ctx.state.borrow_mut();
            let len = vec.len();

            let handle = state
                .create_buffer()
                .map_err(|_| BufferError::cannot_create())?;

            Self::bind(&mut state, handle, target)?;

            let bytes = mem::size_of::<T>() * len;
            let data = slice::from_raw_parts(vec.as_ptr() as *const _, bytes);
            state
                .ctx
                .buffer_data_u8_slice(target, data, glow::STREAM_DRAW);

            let gl_buf = BufferWrapper {
                handle,
                state: ctx.state.clone(),
            };

            Ok(Buffer { gl_buf, buf: vec })
        }
    }

    /// Bind a buffer to a given state regarding the input target.
    fn bind(state: &mut GlowState, handle: glow::Buffer, target: u32) -> Result<(), BufferError> {
        // depending on the buffer target, we are not going to bind it the same way, as the first bind
        // is actually meaningful in WebGL2
        match target {
            glow::ARRAY_BUFFER => state.bind_array_buffer(Some(handle), Bind::Forced),
            glow::ELEMENT_ARRAY_BUFFER => {
                state.bind_element_array_buffer(Some(handle), Bind::Forced)
            }

            // a bit opaque but should never happen
            _ => return Err(BufferError::CannotCreate),
        }

        Ok(())
    }

    pub(crate) fn handle(&self) -> glow::Buffer {
        self.gl_buf.handle
    }
}

unsafe impl<T> BufferBackend<T> for GlowBackend
where
    T: Copy,
{
    type BufferRepr = Buffer<T>;

    unsafe fn new_buffer(&mut self, len: usize) -> Result<Self::BufferRepr, BufferError>
    where
        T: Default,
    {
        Buffer::<T>::new(self, len, T::default(), glow::ARRAY_BUFFER)
    }

    unsafe fn len(buffer: &Self::BufferRepr) -> usize {
        buffer.buf.len()
    }

    unsafe fn from_vec(&mut self, vec: Vec<T>) -> Result<Self::BufferRepr, BufferError> {
        Buffer::from_vec(self, vec, glow::ARRAY_BUFFER)
    }

    unsafe fn repeat(&mut self, len: usize, value: T) -> Result<Self::BufferRepr, BufferError> {
        Buffer::<T>::new(self, len, value, glow::ARRAY_BUFFER)
    }

    unsafe fn at(buffer: &Self::BufferRepr, i: usize) -> Option<T> {
        buffer.buf.get(i).copied()
    }

    unsafe fn whole(buffer: &Self::BufferRepr) -> Vec<T> {
        buffer.buf.iter().copied().collect()
    }

    unsafe fn set(buffer: &mut Self::BufferRepr, i: usize, x: T) -> Result<(), BufferError> {
        let buffer_len = buffer.buf.len();

        if i >= buffer_len {
            Err(BufferError::overflow(i, buffer_len))
        } else {
            // update the cache first
            buffer.buf[i] = x;

            // then update the WebGL buffer
            let mut state = buffer.gl_buf.state.borrow_mut();
            let bytes = mem::size_of::<T>() * buffer_len;
            update_webgl_buffer(
                &mut state,
                buffer.gl_buf.handle,
                buffer.buf.as_ptr() as *const u8,
                bytes,
                i,
            );

            Ok(())
        }
    }

    unsafe fn write_whole(buffer: &mut Self::BufferRepr, values: &[T]) -> Result<(), BufferError> {
        let len = values.len();
        let buffer_len = buffer.buf.len();

        // error if we don’t pass the right number of items
        match len.cmp(&buffer_len) {
            Ordering::Less => return Err(BufferError::too_few_values(len, buffer_len)),

            Ordering::Greater => return Err(BufferError::too_many_values(len, buffer_len)),

            _ => (),
        }

        // update the internal representation of the vector; we clear it first then we extend with
        // the input slice to re-use the allocated region
        buffer.buf.clear();
        buffer.buf.extend_from_slice(values);

        // update the data on GPU
        let mut state = buffer.gl_buf.state.borrow_mut();
        let bytes = mem::size_of::<T>() * buffer_len;
        update_webgl_buffer(
            &mut state,
            buffer.gl_buf.handle,
            buffer.buf.as_ptr() as *const u8,
            bytes,
            0,
        );

        Ok(())
    }

    unsafe fn clear(buffer: &mut Self::BufferRepr, x: T) -> Result<(), BufferError> {
        // copy the value everywhere in the buffer, then simply update the WebGL buffer
        for item in &mut buffer.buf {
            *item = x;
        }

        let mut state = buffer.gl_buf.state.borrow_mut();
        let bytes = buffer.buf.len() * mem::size_of::<T>();
        update_webgl_buffer(
            &mut state,
            buffer.gl_buf.handle,
            buffer.buf.as_ptr() as *const u8,
            bytes,
            0,
        );

        Ok(())
    }
}

pub struct BufferSlice<T> {
    handle: glow::Buffer,
    ptr: *const T,
    len: usize,
    state: Rc<RefCell<GlowState>>,
}

impl BufferSlice<u8> {
    /// Transmute to another type.
    ///
    /// This method is highly unsafe and should only be used when certain the target type is the
    /// one actually represented by the raw bytes.
    pub(crate) unsafe fn transmute<T>(self) -> BufferSlice<T> {
        let handle = self.handle;
        let ptr = self.ptr as *const T;
        let len = self.len / mem::size_of::<T>();
        let state = self.state;

        BufferSlice {
            handle,
            ptr,
            len,
            state,
        }
    }
}

impl<T> Deref for BufferSlice<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// Buffer mutable slice wrapper.
///
/// When a buffer is mapped, we are the only owner of it. We can then read or write from/to the
/// mapped buffer, and then update the GPU buffer on the [`Drop`] implementation.
pub struct BufferSliceMutWrapper {
    handle: glow::Buffer,
    ptr: *mut u8,
    bytes: usize,
    state: Rc<RefCell<GlowState>>,
}

impl Drop for BufferSliceMutWrapper {
    fn drop(&mut self) {
        let mut state = self.state.borrow_mut();
        update_webgl_buffer(&mut state, self.handle, self.ptr, self.bytes, 0);
    }
}

pub struct BufferSliceMut<T> {
    raw: BufferSliceMutWrapper,
    _phantom: PhantomData<T>,
}

impl BufferSliceMut<u8> {
    /// Transmute to another type.
    ///
    /// This method is highly unsafe and should only be used when certain the target type is the
    /// one actually represented by the raw bytes.
    pub(crate) unsafe fn transmute<T>(self) -> BufferSliceMut<T> {
        BufferSliceMut {
            raw: self.raw,
            _phantom: PhantomData,
        }
    }
}

impl<T> Deref for BufferSliceMut<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe {
            slice::from_raw_parts(
                self.raw.ptr as *const T,
                self.raw.bytes / mem::size_of::<T>(),
            )
        }
    }
}

impl<T> DerefMut for BufferSliceMut<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            slice::from_raw_parts_mut(self.raw.ptr as *mut T, self.raw.bytes / mem::size_of::<T>())
        }
    }
}

unsafe impl<T> BufferSliceBackend<T> for GlowBackend
where
    T: Copy,
{
    type SliceRepr = BufferSlice<T>;

    type SliceMutRepr = BufferSliceMut<T>;

    unsafe fn slice_buffer(buffer: &Self::BufferRepr) -> Result<Self::SliceRepr, BufferError> {
        let slice = BufferSlice {
            handle: buffer.gl_buf.handle.clone(),
            ptr: buffer.buf.as_ptr(),
            len: buffer.buf.len(),
            state: buffer.gl_buf.state.clone(),
        };

        Ok(slice)
    }

    unsafe fn slice_buffer_mut(
        buffer: &mut Self::BufferRepr,
    ) -> Result<Self::SliceMutRepr, BufferError> {
        let raw = BufferSliceMutWrapper {
            handle: buffer.gl_buf.handle.clone(),
            ptr: buffer.buf.as_mut_ptr() as *mut u8,
            bytes: buffer.buf.len() * mem::size_of::<T>(),
            state: buffer.gl_buf.state.clone(),
        };
        let slice = BufferSliceMut {
            raw,
            _phantom: PhantomData,
        };

        Ok(slice)
    }
}

/// Update a WebGL buffer by copying an input slice.
fn update_webgl_buffer(
    state: &mut GlowState,
    gl_buf: glow::Buffer,
    data: *const u8,
    bytes: usize,
    offset: usize,
) {
    unsafe {
        state.bind_array_buffer(Some(gl_buf), Bind::Cached);

        let data = slice::from_raw_parts(data as _, bytes);
        state
            .ctx
            .buffer_sub_data_u8_slice(glow::ARRAY_BUFFER, offset as _, data);
    }
}
