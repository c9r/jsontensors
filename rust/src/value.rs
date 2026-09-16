//! The document model: JSON values with tensors in the arrays' places.
//!
//! A document is a JSON object whose values are JSON values or tensors.
//! [`Value`] is generic over what stands in a tensor's place, because a
//! document read whole holds a [`Tensor`] with its bytes and a document read
//! from its header alone holds a [`TensorRef`] naming where the bytes are.

use std::borrow::Cow;

use bytes::Bytes;
use indexmap::IndexMap;

use crate::dtype::Dtype;
use crate::error::Error;

/// A JSON object, with property order preserved.
pub type Map<T> = IndexMap<String, Value<T>>;

/// A document: the object at the root of a file.
pub type Document<T> = Map<T>;

/// A JSON value, or a tensor standing where an array belongs.
#[derive(Clone, Debug, PartialEq)]
pub enum Value<T> {
    Null,
    Bool(bool),
    /// Every JSON number is a double, and the encoder refuses one that is not exactly representable.
    Number(f64),
    String(String),
    Array(Vec<Value<T>>),
    Object(Map<T>),
    Tensor(T),
}

impl<T> Value<T> {
    /// Rebuilds the value with every tensor replaced through `f`, in document order.
    pub fn map_tensors<U, F>(self, f: &mut F) -> Result<Value<U>, Error>
    where
        F: FnMut(T) -> Result<U, Error>,
    {
        Ok(match self {
            Value::Null => Value::Null,
            Value::Bool(b) => Value::Bool(b),
            Value::Number(n) => Value::Number(n),
            Value::String(s) => Value::String(s),
            Value::Array(items) => {
                Value::Array(items.into_iter().map(|item| item.map_tensors(f)).collect::<Result<_, _>>()?)
            }
            Value::Object(map) => Value::Object(map_tensors_in(map, f)?),
            Value::Tensor(tensor) => Value::Tensor(f(tensor)?),
        })
    }
}

/// Rebuilds a map with every tensor replaced through `f`, in document order.
pub fn map_tensors_in<T, U, F>(map: Map<T>, f: &mut F) -> Result<Map<U>, Error>
where
    F: FnMut(T) -> Result<U, Error>,
{
    map.into_iter().map(|(key, value)| Ok((key, value.map_tensors(f)?))).collect()
}

impl<T> From<bool> for Value<T> {
    fn from(b: bool) -> Self {
        Value::Bool(b)
    }
}

impl<T> From<f64> for Value<T> {
    fn from(n: f64) -> Self {
        Value::Number(n)
    }
}

impl<T> From<i32> for Value<T> {
    fn from(n: i32) -> Self {
        Value::Number(n as f64)
    }
}

impl<T> From<&str> for Value<T> {
    fn from(s: &str) -> Self {
        Value::String(s.to_string())
    }
}

impl<T> From<String> for Value<T> {
    fn from(s: String) -> Self {
        Value::String(s)
    }
}

impl<T> From<Vec<Value<T>>> for Value<T> {
    fn from(items: Vec<Value<T>>) -> Self {
        Value::Array(items)
    }
}

impl<T> From<Map<T>> for Value<T> {
    fn from(map: Map<T>) -> Self {
        Value::Object(map)
    }
}

impl From<Tensor> for Value<Tensor> {
    fn from(tensor: Tensor) -> Self {
        Value::Tensor(tensor)
    }
}

/// A reference: where a tensor's bytes are, without the bytes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorRef {
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    /// The first byte, counted from the start of the buffer.
    pub offset: u64,
    /// The byte count, which is the element count times the dtype's width.
    pub length: u64,
}

impl TensorRef {
    /// The element count, which is one for a scalar.
    pub fn elements(&self) -> u64 {
        element_count(&self.shape)
    }

    /// The byte after the last, counted from the start of the buffer.
    pub fn end(&self) -> u64 {
        self.offset + self.length
    }
}

/// A tensor with its bytes, which are a view when the source allows one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    /// Row-major little-endian bytes, exactly the element count times the width.
    pub data: Bytes,
}

/// The element count of a shape, which is one for the empty shape.
pub fn element_count(shape: &[u64]) -> u64 {
    shape.iter().product()
}

impl Tensor {
    /// A tensor over bytes that must already be the shape's byte count.
    pub fn new(dtype: Dtype, shape: Vec<u64>, data: impl Into<Bytes>) -> Result<Tensor, Error> {
        let data = data.into();
        let expected = element_count(&shape) * dtype.width();
        if data.len() as u64 != expected {
            return Err(Error::TensorLength { dtype, shape, length: data.len() as u64, expected });
        }
        Ok(Tensor { dtype, shape, data })
    }

    /// A tensor copied from a slice of elements, in row-major order.
    pub fn from_slice<E: Element>(shape: Vec<u64>, elements: &[E]) -> Result<Tensor, Error> {
        Tensor::new(E::DTYPE, shape, bytemuck::cast_slice::<E, u8>(elements).to_vec())
    }

    /// A rank-one tensor copied from a slice of elements.
    pub fn from_vec<E: Element>(elements: Vec<E>) -> Tensor {
        Tensor::from_slice(vec![elements.len() as u64], &elements).expect("a vector's length is its shape")
    }

    /// The element count.
    pub fn elements(&self) -> u64 {
        element_count(&self.shape)
    }

    /// The reference this tensor would have at an offset.
    pub fn reference(&self, offset: u64) -> TensorRef {
        TensorRef { dtype: self.dtype, shape: self.shape.clone(), offset, length: self.data.len() as u64 }
    }

    /// The elements as a typed slice over the bytes, without copying.
    ///
    /// Fails when the dtype is not `E`, or when the bytes are not aligned for
    /// `E`, which cannot happen for a mapped file and can for a buffer that
    /// arrived over a wire. [`Tensor::elements_of`] handles both.
    pub fn as_slice<E: Element>(&self) -> Result<&[E], Error> {
        if self.dtype != E::DTYPE {
            return Err(Error::DtypeMismatch { expected: E::DTYPE, actual: self.dtype });
        }
        bytemuck::try_cast_slice(&self.data).map_err(|_| Error::Misaligned { dtype: self.dtype })
    }

    /// The elements as a typed slice, viewing the bytes when they are aligned and copying when they are not.
    pub fn elements_of<E: Element>(&self) -> Result<Cow<'_, [E]>, Error> {
        if self.dtype != E::DTYPE {
            return Err(Error::DtypeMismatch { expected: E::DTYPE, actual: self.dtype });
        }
        Ok(match bytemuck::try_cast_slice(&self.data) {
            Ok(slice) => Cow::Borrowed(slice),
            Err(_) => Cow::Owned(bytemuck::pod_collect_to_vec(&self.data)),
        })
    }

    /// The elements copied into a vector.
    pub fn to_vec<E: Element>(&self) -> Result<Vec<E>, Error> {
        self.elements_of::<E>().map(Cow::into_owned)
    }

    /// The elements of a `BOOL` tensor, refusing any byte that is not 0 or 1.
    pub fn to_bools(&self) -> Result<Vec<bool>, Error> {
        if self.dtype != Dtype::Bool {
            return Err(Error::DtypeMismatch { expected: Dtype::Bool, actual: self.dtype });
        }
        self.data
            .iter()
            .map(|&byte| match byte {
                0 => Ok(false),
                1 => Ok(true),
                other => Err(Error::InvalidBool { byte: other }),
            })
            .collect()
    }
}

/// A Rust type that is the natural form of one dtype.
///
/// `bool` is not one, because not every byte is a valid `bool`, so `BOOL`
/// tensors read through [`Tensor::to_bools`] or as `u8`.
pub trait Element: bytemuck::Pod {
    const DTYPE: Dtype;
}

macro_rules! element {
    ($($ty:ty => $dtype:expr),* $(,)?) => {
        $(impl Element for $ty { const DTYPE: Dtype = $dtype; })*
    };
}

element! {
    f64 => Dtype::F64,
    f32 => Dtype::F32,
    half::f16 => Dtype::F16,
    half::bf16 => Dtype::BF16,
    i64 => Dtype::I64,
    i32 => Dtype::I32,
    i16 => Dtype::I16,
    i8 => Dtype::I8,
    u64 => Dtype::U64,
    u32 => Dtype::U32,
    u16 => Dtype::U16,
    u8 => Dtype::U8,
}

/// A `BOOL` tensor from bools.
pub fn bools(shape: Vec<u64>, values: &[bool]) -> Result<Tensor, Error> {
    Tensor::new(Dtype::Bool, shape, values.iter().map(|&b| b as u8).collect::<Vec<u8>>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensors_check_their_length() {
        assert!(Tensor::new(Dtype::F32, vec![2, 2], vec![0u8; 16]).is_ok());
        assert!(Tensor::new(Dtype::F32, vec![2, 2], vec![0u8; 15]).is_err());
        assert!(Tensor::new(Dtype::U8, vec![], vec![7u8]).is_ok(), "an empty shape is a scalar");
        assert!(Tensor::new(Dtype::I64, vec![0, 5], Vec::new()).is_ok(), "a zero in the shape is no elements");
    }

    #[test]
    fn slices_view_and_copy() {
        let tensor = Tensor::from_slice(vec![2, 2], &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(tensor.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert!(tensor.as_slice::<i32>().is_err());
        assert_eq!(tensor.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        let misaligned = Tensor::new(Dtype::F32, vec![1], Bytes::from(vec![9u8, 0, 0, 128, 63]).slice(1..5)).unwrap();
        assert_eq!(misaligned.elements_of::<f32>().unwrap().as_ref(), &[1.0f32]);
    }

    #[test]
    fn bools_are_strict() {
        let tensor = bools(vec![3], &[true, false, true]).unwrap();
        assert_eq!(tensor.to_bools().unwrap(), vec![true, false, true]);
        let bad = Tensor::new(Dtype::Bool, vec![1], vec![2u8]).unwrap();
        assert!(bad.to_bools().is_err());
    }
}
