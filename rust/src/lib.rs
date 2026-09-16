//! jsontensors keeps a JSON document's arrays out of the text and in a binary
//! buffer behind it, with each array in its place in the document and no copy
//! on read.
//!
//! A file is an eight-byte little-endian header length, that many bytes of
//! UTF-8 JSON padded to an eight-byte boundary, and then the tensors' bytes.
//! In the JSON, each array's place holds a reference naming its dtype, shape,
//! and byte range. This crate transcribes the specification in the
//! repository's `SPEC.md`.
//!
//! The four operations:
//!
//! - [`read`] maps a file and returns the document with each tensor a view over the mapping.
//! - [`read_header`] returns the document with a [`TensorRef`] in each array's place, touching no tensor bytes.
//! - [`ranged::open`] parses the header once and reads any byte range of any tensor with one positioned read.
//! - [`write`] streams a document to a temporary file beside the target and renames it into place.
//!
//! [`encode`] and [`decode`] do the same in memory, for documents that travel as bytes.
//!
//! ```
//! use jsontensors::{Document, Tensor, Value};
//!
//! let mut document: Document<Tensor> = Document::new();
//! document.insert("title".into(), "Rain on a tin roof".into());
//! document.insert("samples".into(), Tensor::from_vec(vec![0.25f32, -0.5, 1.0]).into());
//!
//! let bytes = jsontensors::encode(&document).unwrap();
//! let decoded = jsontensors::decode(bytes).unwrap();
//! let Value::Tensor(samples) = &decoded["samples"] else { panic!() };
//! assert_eq!(samples.elements_of::<f32>().unwrap().as_ref(), &[0.25, -0.5, 1.0]);
//! ```

#[cfg(target_endian = "big")]
compile_error!("jsontensors views little-endian bytes as native element slices and supports little-endian targets");

pub mod decode;
pub mod dtype;
pub mod encode;
pub mod error;
pub mod header;
pub mod ranged;
pub mod value;

pub use decode::{decode, read, read_header};
pub use dtype::Dtype;
pub use encode::{encode, layout, write, write_to};
pub use error::{Category, Error};
pub use header::{Header, parse_header};
pub use ranged::{Ranged, Source};
pub use value::{Document, Element, Map, Tensor, TensorRef, Value, bools};

/// The file suffix.
pub const SUFFIX: &str = ".jsontensors";

/// The media type.
pub const MEDIA_TYPE: &str = "application/x-jsontensors";
