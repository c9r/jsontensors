//! Parsing the header: the length, the JSON, and the references, validated against the buffer.

use serde_json::Value as Json;

use crate::dtype::Dtype;
use crate::error::{Error, Path};
use crate::value::{Document, Map, TensorRef, Value, element_count};

/// The largest integer magnitude a double represents exactly.
pub const MAX_SAFE_INTEGER: i128 = 1 << 53;

/// A parsed header: the document with references in the arrays' places, and where the buffer is.
#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    pub document: Document<TensorRef>,
    /// The header's byte count, padding included, as the length prefix states it.
    pub header_length: u64,
    /// The buffer's first byte, counted from the start of the file, which is eight plus the header length.
    pub buffer_start: u64,
    /// The buffer's byte count, which the references must tile exactly.
    pub buffer_length: u64,
}

/// The header length a prefix declares, checked against the file's total size.
fn declared_length(prefix: &[u8], total: u64) -> Result<u64, Error> {
    if total < 8 {
        return Err(Error::TruncatedLength);
    }
    if prefix.len() < 8 {
        return Err(Error::NeedMore { required: total.min(4096) });
    }
    let length = u64::from_le_bytes(prefix[..8].try_into().expect("eight bytes"));
    if length > total - 8 {
        return Err(Error::TruncatedHeader { length, total });
    }
    Ok(length)
}

/// Parses the header from a prefix of a file whose total size is known.
///
/// A reader with an object store or a socket learns the size from the store,
/// reads a first prefix, and retries with the size an [`Error::NeedMore`]
/// names until the header is in hand. The references validate against the
/// total, so a truncated object fails here exactly as a truncated file does.
pub fn parse_header(prefix: &[u8], total: u64) -> Result<Header, Error> {
    let header_length = declared_length(prefix, total)?;
    let end = 8 + header_length;
    if (prefix.len() as u64) < end {
        return Err(Error::NeedMore { required: end });
    }
    let buffer_length = total - end;
    let document = parse_json(&prefix[8..end as usize], buffer_length)?;
    Ok(Header { document, header_length, buffer_start: end, buffer_length })
}

/// Parses the header JSON and validates its references against a buffer of the given length.
pub fn parse_json(blob: &[u8], buffer_length: u64) -> Result<Document<TensorRef>, Error> {
    let raw: Json = serde_json::from_slice(blob).map_err(|e| Error::InvalidJson(e.to_string()))?;
    let Json::Object(object) = raw else {
        return Err(Error::NonObjectRoot);
    };
    if object.contains_key("$dtype") {
        return Err(Error::NonObjectRoot);
    }
    let mut references = Vec::new();
    let mut path = Path::root();
    let document = convert_object(object, &mut path, &mut references)?;
    check_tiling(&references, buffer_length)?;
    Ok(document)
}

fn convert(raw: Json, path: &mut Path, references: &mut Vec<TensorRef>) -> Result<Value<TensorRef>, Error> {
    Ok(match raw {
        Json::Null => Value::Null,
        Json::Bool(b) => Value::Bool(b),
        Json::Number(n) => Value::Number(number(&n, path)?),
        Json::String(s) => Value::String(s),
        Json::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for (index, item) in items.into_iter().enumerate() {
                path.push_index(index);
                out.push(convert(item, path, references)?);
                path.pop();
            }
            Value::Array(out)
        }
        Json::Object(object) => {
            if object.contains_key("$dtype") {
                let reference = parse_reference(&object, path)?;
                references.push(reference.clone());
                Value::Tensor(reference)
            } else {
                Value::Object(convert_object(object, path, references)?)
            }
        }
    })
}

fn convert_object(
    object: serde_json::Map<String, Json>,
    path: &mut Path,
    references: &mut Vec<TensorRef>,
) -> Result<Map<TensorRef>, Error> {
    let mut out = Map::with_capacity(object.len());
    for (key, item) in object {
        let name = if let Some(rest) = key.strip_prefix("$$") {
            format!("${rest}")
        } else if key.starts_with('$') {
            return Err(Error::StrayDollar { path: path.to_string(), name: key });
        } else {
            key.clone()
        };
        path.push_key(&key);
        let value = convert(item, path, references)?;
        path.pop();
        out.insert(name, value);
    }
    Ok(out)
}

/// A JSON number as the double it exactly represents, refusing one that is not.
fn number(n: &serde_json::Number, path: &Path) -> Result<f64, Error> {
    let literal = n.to_string();
    let refuse = || Error::Number { path: path.to_string(), literal: literal.clone() };
    let is_integer_literal = literal.bytes().all(|b| b.is_ascii_digit() || b == b'-');
    if is_integer_literal {
        let value: i128 = literal.parse().map_err(|_| refuse())?;
        if value.abs() > MAX_SAFE_INTEGER {
            return Err(refuse());
        }
        return Ok(value as f64);
    }
    let value: f64 = literal.parse().map_err(|_| refuse())?;
    if !value.is_finite() {
        return Err(refuse());
    }
    Ok(value)
}

const REFERENCE_KEYS: [&str; 4] = ["$dtype", "shape", "offset", "length"];

fn parse_reference(object: &serde_json::Map<String, Json>, path: &Path) -> Result<TensorRef, Error> {
    let malformed = |reason: &str| Error::MalformedReference { path: path.to_string(), reason: reason.to_string() };
    if object.len() != 4 || !REFERENCE_KEYS.iter().all(|key| object.contains_key(*key)) {
        let keys: Vec<&str> = object.keys().map(String::as_str).collect();
        return Err(malformed(&format!("its properties are {keys:?}")));
    }
    let dtype = match &object["$dtype"] {
        Json::String(name) => Dtype::from_name(name).ok_or_else(|| malformed(&format!("{name:?} is not a dtype")))?,
        _ => return Err(malformed("$dtype is not a string")),
    };
    let shape = match &object["shape"] {
        Json::Array(items) => items
            .iter()
            .map(|item| {
                item.as_number()
                    .and_then(non_negative_integer)
                    .ok_or_else(|| malformed("shape is not a list of non-negative integers"))
            })
            .collect::<Result<Vec<u64>, Error>>()?,
        _ => return Err(malformed("shape is not a list")),
    };
    let offset = object["offset"]
        .as_number()
        .and_then(non_negative_integer)
        .ok_or_else(|| malformed("offset is not a non-negative integer"))?;
    let length = object["length"]
        .as_number()
        .and_then(non_negative_integer)
        .ok_or_else(|| malformed("length is not a non-negative integer"))?;
    let expected = element_count(&shape)
        .checked_mul(dtype.width())
        .ok_or_else(|| malformed("the shape's byte count overflows"))?;
    if length != expected {
        return Err(Error::LengthMismatch { path: path.to_string(), length, expected });
    }
    Ok(TensorRef { dtype, shape, offset, length })
}

/// A number that is an integer literal within the safe range and not negative.
fn non_negative_integer(n: &serde_json::Number) -> Option<u64> {
    let literal = n.to_string();
    if !literal.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let value: i128 = literal.parse().ok()?;
    (value <= MAX_SAFE_INTEGER).then_some(value as u64)
}

/// Checks that the references, ordered by offset, tile the buffer exactly.
pub fn check_tiling(references: &[TensorRef], buffer_length: u64) -> Result<(), Error> {
    let mut ranges: Vec<(u64, u64)> = references.iter().map(|r| (r.offset, r.end())).collect();
    ranges.sort_unstable();
    let mut cursor = 0u64;
    for (begin, end) in ranges {
        if begin != cursor {
            return Err(Error::Tiling(if begin > cursor {
                format!("a gap of {} bytes before offset {begin}", begin - cursor)
            } else {
                format!("the reference at offset {begin} overlaps the one before it")
            }));
        }
        cursor = end;
    }
    if cursor != buffer_length {
        return Err(Error::Tiling(format!("the references end at {cursor}, but the buffer is {buffer_length} bytes")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Category;

    fn header_bytes(json: &str) -> Vec<u8> {
        let mut blob = json.as_bytes().to_vec();
        while !(8 + blob.len()).is_multiple_of(8) {
            blob.push(b' ');
        }
        let mut out = (blob.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&blob);
        out
    }

    fn parse(json: &str, buffer_length: u64) -> Result<Header, Error> {
        let head = header_bytes(json);
        parse_header(&head, head.len() as u64 + buffer_length)
    }

    fn category(json: &str, buffer_length: u64) -> Category {
        parse(json, buffer_length).unwrap_err().category().expect("a validity error")
    }

    #[test]
    fn parses_a_document_with_references() {
        let header = parse(r#"{"a":{"$dtype":"F32","shape":[2],"offset":4,"length":8},"b":{"$dtype":"U8","shape":[4],"offset":0,"length":4},"$$k":1}"#, 12).unwrap();
        assert_eq!(header.buffer_length, 12);
        let Value::Tensor(a) = &header.document["a"] else { panic!("a is a tensor") };
        assert_eq!(a, &TensorRef { dtype: Dtype::F32, shape: vec![2], offset: 4, length: 8 });
        assert_eq!(header.document["$k"], Value::Number(1.0));
    }

    #[test]
    fn prefixes_ask_for_more_until_the_header_is_whole() {
        let head = header_bytes(r#"{"x":1}"#);
        let total = head.len() as u64;
        assert!(matches!(parse_header(&head[..3], total), Err(Error::NeedMore { required }) if required == total));
        assert!(matches!(parse_header(&head[..10], total), Err(Error::NeedMore { required }) if required == total));
        assert!(parse_header(&head, total).is_ok());
    }

    #[test]
    fn categorizes_invalid_files() {
        assert!(matches!(parse_header(&[1, 2, 3], 3), Err(Error::TruncatedLength)));
        assert!(matches!(parse_header(&100u64.to_le_bytes(), 50), Err(Error::TruncatedHeader { .. })));
        assert_eq!(category("{\"a\":", 0), Category::InvalidJson);
        assert_eq!(category("[1]", 0), Category::NonObjectRoot);
        assert_eq!(category(r#"{"$dtype":"U8","shape":[1],"offset":0,"length":1}"#, 1), Category::NonObjectRoot);
        assert_eq!(category(r#"{"$x":1}"#, 0), Category::StrayDollar);
        assert_eq!(category(r#"{"a":[{"$y":1}]}"#, 0), Category::StrayDollar);
        assert_eq!(
            category(r#"{"a":{"$dtype":"F128","shape":[1],"offset":0,"length":16}}"#, 16),
            Category::MalformedReference
        );
        assert_eq!(category(r#"{"a":{"$dtype":"U8","shape":[1],"offset":0}}"#, 1), Category::MalformedReference);
        assert_eq!(
            category(r#"{"a":{"$dtype":"U8","shape":[-1],"offset":0,"length":1}}"#, 1),
            Category::MalformedReference
        );
        assert_eq!(
            category(r#"{"a":{"$dtype":"U8","shape":[1.5],"offset":0,"length":1}}"#, 1),
            Category::MalformedReference
        );
        assert_eq!(category(r#"{"a":{"$dtype":"U8","shape":[2],"offset":0,"length":1}}"#, 1), Category::LengthMismatch);
        assert_eq!(category(r#"{"a":{"$dtype":"U8","shape":[2],"offset":1,"length":2}}"#, 3), Category::Tiling);
        assert_eq!(
            category(
                r#"{"a":{"$dtype":"U8","shape":[2],"offset":0,"length":2},"b":{"$dtype":"U8","shape":[2],"offset":1,"length":2}}"#,
                3
            ),
            Category::Tiling
        );
        assert_eq!(category(r#"{"a":{"$dtype":"U8","shape":[2],"offset":0,"length":2}}"#, 3), Category::Tiling);
        assert_eq!(category(r#"{"n":9007199254740993}"#, 0), Category::Number);
        assert_eq!(category(r#"{"n":1e400}"#, 0), Category::Number);
    }

    #[test]
    fn accepts_the_edges() {
        assert!(parse(r#"{"n":9007199254740992,"m":-9007199254740992}"#, 0).is_ok());
        assert!(
            parse(r#"{"s":{"$dtype":"F64","shape":[],"offset":0,"length":8}}"#, 8).is_ok(),
            "an empty shape is a scalar"
        );
        assert!(
            parse(r#"{"e":{"$dtype":"F64","shape":[0,3],"offset":0,"length":0}}"#, 0).is_ok(),
            "a zero shape is no bytes"
        );
        assert!(parse(r#"{"a":{"$dtype":"U8","shape":[2],"offset":0,"length":2},"e":{"$dtype":"U8","shape":[0],"offset":2,"length":0}}"#, 2).is_ok());
        assert_eq!(
            category(
                r#"{"a":{"$dtype":"U8","shape":[2],"offset":0,"length":2},"e":{"$dtype":"U8","shape":[0],"offset":1,"length":0}}"#,
                2
            ),
            Category::Tiling
        );
    }
}
