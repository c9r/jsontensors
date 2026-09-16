//! Errors, each carrying the conformance category it belongs to.

use std::fmt;

use crate::dtype::Dtype;

/// The categories the conformance suite names. A decoding error belongs to exactly one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Category {
    TruncatedLength,
    TruncatedHeader,
    InvalidJson,
    NonObjectRoot,
    MalformedReference,
    LengthMismatch,
    StrayDollar,
    Tiling,
    Number,
}

impl Category {
    /// The category's name in a conformance case.
    pub fn name(self) -> &'static str {
        match self {
            Category::TruncatedLength => "truncated-length",
            Category::TruncatedHeader => "truncated-header",
            Category::InvalidJson => "invalid-json",
            Category::NonObjectRoot => "non-object-root",
            Category::MalformedReference => "malformed-reference",
            Category::LengthMismatch => "length-mismatch",
            Category::StrayDollar => "stray-dollar",
            Category::Tiling => "tiling",
            Category::Number => "number",
        }
    }
}

impl fmt::Display for Category {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("the file ends before the eight length bytes")]
    TruncatedLength,

    #[error("the header length {length} overruns the {total}-byte file")]
    TruncatedHeader { length: u64, total: u64 },

    /// A prefix ended inside the header of a file that does carry it whole. Retry with at least `required` bytes.
    #[error("the header needs {required} bytes")]
    NeedMore { required: u64 },

    #[error("the header is not UTF-8 JSON: {0}")]
    InvalidJson(String),

    #[error("the header's root is not an object")]
    NonObjectRoot,

    #[error("the reference at {path} is malformed: {reason}")]
    MalformedReference { path: String, reason: String },

    #[error("the reference at {path} has length {length}, but its shape and dtype make {expected} bytes")]
    LengthMismatch { path: String, length: u64, expected: u64 },

    #[error("the property {name:?} at {path} has a single leading $ outside a reference")]
    StrayDollar { path: String, name: String },

    #[error("the references do not tile the buffer: {0}")]
    Tiling(String),

    #[error("the number {literal} at {path} is not exactly representable as a double")]
    Number { path: String, literal: String },

    #[error("a tensor of dtype {dtype} and shape {shape:?} takes {expected} bytes, not {length}")]
    TensorLength { dtype: Dtype, shape: Vec<u64>, length: u64, expected: u64 },

    #[error("the number at {path} is not finite, and JSON cannot spell it")]
    NonFinite { path: String },

    #[error("the number at {path} is an integer beyond 2^53, which is not exactly representable")]
    IntegerRange { path: String },

    #[error("the tensor is {actual}, not {expected}")]
    DtypeMismatch { expected: Dtype, actual: Dtype },

    #[error("the tensor's {dtype} bytes are not aligned for a typed view; copy them instead")]
    Misaligned { dtype: Dtype },

    #[error("a BOOL tensor holds the byte {byte}, which is neither 0 nor 1")]
    InvalidBool { byte: u8 },

    #[error("the range [{start}, {end}) lies outside a {length}-byte reference")]
    RangeOutside { start: u64, end: u64, length: u64 },

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl Error {
    /// The conformance category of a decoding error, `None` for errors that are not about a file's validity.
    pub fn category(&self) -> Option<Category> {
        Some(match self {
            Error::TruncatedLength => Category::TruncatedLength,
            Error::TruncatedHeader { .. } => Category::TruncatedHeader,
            Error::InvalidJson(_) => Category::InvalidJson,
            Error::NonObjectRoot => Category::NonObjectRoot,
            Error::MalformedReference { .. } => Category::MalformedReference,
            Error::LengthMismatch { .. } => Category::LengthMismatch,
            Error::StrayDollar { .. } => Category::StrayDollar,
            Error::Tiling(_) => Category::Tiling,
            Error::Number { .. } => Category::Number,
            _ => return None,
        })
    }
}

/// A path into a document, kept as segments and printed as a JSON Pointer only when an error needs it.
#[derive(Clone, Debug, Default)]
pub struct Path {
    segments: Vec<Segment>,
}

#[derive(Clone, Debug)]
enum Segment {
    Key(String),
    Index(usize),
}

impl Path {
    pub fn root() -> Path {
        Path::default()
    }

    pub fn push_key(&mut self, key: &str) {
        self.segments.push(Segment::Key(key.to_string()));
    }

    pub fn push_index(&mut self, index: usize) {
        self.segments.push(Segment::Index(index));
    }

    pub fn pop(&mut self) {
        self.segments.pop();
    }

    /// The path as a JSON Pointer, `/a/0/b`, with `~` and `/` escaped as the pointer grammar requires.
    pub fn pointer(&self) -> String {
        let mut out = String::new();
        for segment in &self.segments {
            out.push('/');
            match segment {
                Segment::Key(key) => out.push_str(&key.replace('~', "~0").replace('/', "~1")),
                Segment::Index(index) => out.push_str(&index.to_string()),
            }
        }
        out
    }
}

impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pointer = self.pointer();
        f.write_str(if pointer.is_empty() { "the root" } else { &pointer })
    }
}
