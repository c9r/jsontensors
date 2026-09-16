//! The dtype table: the element types a tensor can hold.

use std::fmt;

/// An element type. Every multi-byte value is little-endian in the buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Dtype {
    /// IEEE 754 binary64.
    F64,
    /// IEEE 754 binary32.
    F32,
    /// IEEE 754 binary16.
    F16,
    /// bfloat16, the high half of a binary32.
    BF16,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    /// One byte per element, 0 or 1.
    Bool,
}

impl Dtype {
    /// Every dtype, in the order the table lists them.
    pub const ALL: [Dtype; 13] = [
        Dtype::F64,
        Dtype::F32,
        Dtype::F16,
        Dtype::BF16,
        Dtype::I64,
        Dtype::I32,
        Dtype::I16,
        Dtype::I8,
        Dtype::U64,
        Dtype::U32,
        Dtype::U16,
        Dtype::U8,
        Dtype::Bool,
    ];

    /// The name a reference's `$dtype` carries.
    pub fn name(self) -> &'static str {
        match self {
            Dtype::F64 => "F64",
            Dtype::F32 => "F32",
            Dtype::F16 => "F16",
            Dtype::BF16 => "BF16",
            Dtype::I64 => "I64",
            Dtype::I32 => "I32",
            Dtype::I16 => "I16",
            Dtype::I8 => "I8",
            Dtype::U64 => "U64",
            Dtype::U32 => "U32",
            Dtype::U16 => "U16",
            Dtype::U8 => "U8",
            Dtype::Bool => "BOOL",
        }
    }

    /// The dtype a `$dtype` value names, if it names one.
    pub fn from_name(name: &str) -> Option<Dtype> {
        Dtype::ALL.into_iter().find(|dtype| dtype.name() == name)
    }

    /// The width of one element in bytes.
    pub fn width(self) -> u64 {
        match self {
            Dtype::F64 | Dtype::I64 | Dtype::U64 => 8,
            Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
            Dtype::F16 | Dtype::BF16 | Dtype::I16 | Dtype::U16 => 2,
            Dtype::I8 | Dtype::U8 | Dtype::Bool => 1,
        }
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        for dtype in Dtype::ALL {
            assert_eq!(Dtype::from_name(dtype.name()), Some(dtype));
        }
        assert_eq!(Dtype::from_name("F128"), None);
        assert_eq!(Dtype::from_name("bool"), None);
    }

    #[test]
    fn widths_match_the_table() {
        assert_eq!(Dtype::F64.width(), 8);
        assert_eq!(Dtype::BF16.width(), 2);
        assert_eq!(Dtype::Bool.width(), 1);
    }
}
