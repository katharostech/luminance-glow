//! A small helper module to work with slices.

/// A small helper to flatten a slice of arrays. Given a &[[T; N]], it returns a &[T], because
/// there is no way to flatten slices of arrays in the standard library.
///
/// Please do not abuse.
macro_rules! flatten_slice {
  ($e:ident : $t:ty , len = $len:expr) => {
    std::slice::from_raw_parts($e.as_ptr() as *const $t, $len)
  };
}
