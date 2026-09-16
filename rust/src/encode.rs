//! Encoding: laying tensors out, serializing the header canonically, and writing the file.

use std::io::Write;
use std::path::Path as FilePath;

use crate::error::{Error, Path};
use crate::header::MAX_SAFE_INTEGER;
use crate::value::{Document, Tensor, TensorRef, Value};

/// A document laid out for writing: the head and the tensors in buffer order.
#[derive(Debug)]
pub struct Layout<'a> {
    /// The length prefix and the padded header JSON.
    pub head: Vec<u8>,
    /// The tensors in the order their bytes follow the head.
    pub tensors: Vec<&'a Tensor>,
    /// The buffer's byte count, which is the sum of the tensors' lengths.
    pub buffer_length: u64,
}

impl Layout<'_> {
    /// The whole file's byte count.
    pub fn total_length(&self) -> u64 {
        self.head.len() as u64 + self.buffer_length
    }
}

/// Lays a document out: tensors collected in document order, placed widest
/// dtype first with ties in collection order, and the header serialized with
/// each tensor's reference in its place.
pub fn layout(document: &Document<Tensor>) -> Result<Layout<'_>, Error> {
    let mut tensors = Vec::new();
    let mut path = Path::root();
    for (key, value) in document {
        path.push_key(key);
        collect(value, &mut path, &mut tensors)?;
        path.pop();
    }

    let mut order: Vec<usize> = (0..tensors.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(tensors[i].dtype.width()));
    let mut references =
        vec![TensorRef { dtype: crate::Dtype::U8, shape: Vec::new(), offset: 0, length: 0 }; tensors.len()];
    let mut offset = 0u64;
    for &i in &order {
        references[i] = tensors[i].reference(offset);
        offset += tensors[i].data.len() as u64;
    }

    let mut blob = Vec::new();
    let mut next = references.iter();
    write_object(&mut blob, document, &mut next);
    let padding = (8 - (8 + blob.len()) % 8) % 8;
    blob.resize(blob.len() + padding, b' ');
    let mut head = (blob.len() as u64).to_le_bytes().to_vec();
    head.extend_from_slice(&blob);

    Ok(Layout { head, tensors: order.into_iter().map(|i| tensors[i]).collect(), buffer_length: offset })
}

/// Collects tensors in document order and refuses numbers JSON cannot carry exactly.
fn collect<'a>(value: &'a Value<Tensor>, path: &mut Path, tensors: &mut Vec<&'a Tensor>) -> Result<(), Error> {
    match value {
        Value::Tensor(tensor) => tensors.push(tensor),
        Value::Number(n) => {
            if !n.is_finite() {
                return Err(Error::NonFinite { path: path.to_string() });
            }
        }
        Value::Array(items) => {
            for (index, item) in items.iter().enumerate() {
                path.push_index(index);
                collect(item, path, tensors)?;
                path.pop();
            }
        }
        Value::Object(map) => {
            for (key, item) in map {
                path.push_key(key);
                collect(item, path, tensors)?;
                path.pop();
            }
        }
        Value::Null | Value::Bool(_) | Value::String(_) => {}
    }
    Ok(())
}

fn write_value<'a>(out: &mut Vec<u8>, value: &Value<Tensor>, next: &mut impl Iterator<Item = &'a TensorRef>) {
    match value {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(true) => out.extend_from_slice(b"true"),
        Value::Bool(false) => out.extend_from_slice(b"false"),
        Value::Number(n) => write_number(out, *n),
        Value::String(s) => write_string(out, s),
        Value::Array(items) => {
            out.push(b'[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(b',');
                }
                write_value(out, item, next);
            }
            out.push(b']');
        }
        Value::Object(map) => write_object(out, map, next),
        Value::Tensor(_) => {
            let reference = next.next().expect("one reference per collected tensor");
            write_reference(out, reference);
        }
    }
}

fn write_object<'a>(out: &mut Vec<u8>, map: &Document<Tensor>, next: &mut impl Iterator<Item = &'a TensorRef>) {
    out.push(b'{');
    for (index, (key, value)) in map.iter().enumerate() {
        if index > 0 {
            out.push(b',');
        }
        if key.starts_with('$') {
            write_string(out, &format!("${key}"));
        } else {
            write_string(out, key);
        }
        out.push(b':');
        write_value(out, value, next);
    }
    out.push(b'}');
}

/// Writes a reference in its canonical form and property order.
pub fn write_reference(out: &mut Vec<u8>, reference: &TensorRef) {
    out.extend_from_slice(b"{\"$dtype\":\"");
    out.extend_from_slice(reference.dtype.name().as_bytes());
    out.extend_from_slice(b"\",\"shape\":[");
    for (index, dim) in reference.shape.iter().enumerate() {
        if index > 0 {
            out.push(b',');
        }
        out.extend_from_slice(dim.to_string().as_bytes());
    }
    out.extend_from_slice(b"],\"offset\":");
    out.extend_from_slice(reference.offset.to_string().as_bytes());
    out.extend_from_slice(b",\"length\":");
    out.extend_from_slice(reference.length.to_string().as_bytes());
    out.push(b'}');
}

/// Writes a finite double in its shortest round-trip form.
///
/// An integer-valued double within ±2^53 is written as an integer literal.
/// One beyond that is written in exponent form, so no reader takes it for an
/// integer literal claiming an exactness no double holds. Everything else is
/// the shortest decimal that parses back to the same double.
pub fn write_number(out: &mut Vec<u8>, value: f64) {
    if value.fract() == 0.0 && value.abs() <= MAX_SAFE_INTEGER as f64 {
        out.extend_from_slice((value as i64).to_string().as_bytes());
    } else if value.fract() == 0.0 {
        out.extend_from_slice(format!("{value:e}").as_bytes());
    } else {
        out.extend_from_slice(ryu::Buffer::new().format_finite(value).as_bytes());
    }
}

/// Writes a JSON string with the escapes JSON requires and nothing more, so non-ASCII text stays UTF-8.
pub fn write_string(out: &mut Vec<u8>, text: &str) {
    out.push(b'"');
    for ch in text.chars() {
        match ch {
            '"' => out.extend_from_slice(b"\\\""),
            '\\' => out.extend_from_slice(b"\\\\"),
            '\n' => out.extend_from_slice(b"\\n"),
            '\r' => out.extend_from_slice(b"\\r"),
            '\t' => out.extend_from_slice(b"\\t"),
            '\u{8}' => out.extend_from_slice(b"\\b"),
            '\u{c}' => out.extend_from_slice(b"\\f"),
            c if (c as u32) < 0x20 => out.extend_from_slice(format!("\\u{:04x}", c as u32).as_bytes()),
            c => {
                let mut buf = [0u8; 4];
                out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    out.push(b'"');
}

/// Writes a document to a writer, head first and then each tensor's bytes in buffer order.
pub fn write_to<W: Write>(writer: &mut W, document: &Document<Tensor>) -> Result<(), Error> {
    let layout = layout(document)?;
    writer.write_all(&layout.head)?;
    for tensor in layout.tensors {
        writer.write_all(&tensor.data)?;
    }
    Ok(())
}

/// The whole file as bytes.
pub fn encode(document: &Document<Tensor>) -> Result<Vec<u8>, Error> {
    let layout = layout(document)?;
    let mut out = Vec::with_capacity(layout.total_length() as usize);
    out.extend_from_slice(&layout.head);
    for tensor in layout.tensors {
        out.extend_from_slice(&tensor.data);
    }
    Ok(out)
}

/// Writes a document to a path atomically: a temporary file in the target's
/// directory, synced, then renamed over the target.
pub fn write(path: &FilePath, document: &Document<Tensor>) -> Result<(), Error> {
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| FilePath::new("."));
    let mut temp = tempfile::Builder::new().prefix(".jsontensors-").tempfile_in(dir)?;
    {
        let mut buffered = std::io::BufWriter::new(temp.as_file_mut());
        write_to(&mut buffered, document)?;
        buffered.flush()?;
    }
    temp.as_file().sync_all()?;
    temp.persist(path).map_err(|e| e.error)?;
    Ok(())
}

impl<T> Value<T> {
    /// Whether an encoder would accept this value's numbers.
    pub fn check_numbers(&self) -> Result<(), Error> {
        let mut path = Path::root();
        check_numbers(self, &mut path)
    }
}

fn check_numbers<T>(value: &Value<T>, path: &mut Path) -> Result<(), Error> {
    match value {
        Value::Number(n) if !n.is_finite() => Err(Error::NonFinite { path: path.to_string() }),
        Value::Array(items) => {
            for (index, item) in items.iter().enumerate() {
                path.push_index(index);
                check_numbers(item, path)?;
                path.pop();
            }
            Ok(())
        }
        Value::Object(map) => {
            for (key, item) in map {
                path.push_key(key);
                check_numbers(item, path)?;
                path.pop();
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dtype;
    use crate::value::Map;

    fn number_text(value: f64) -> String {
        let mut out = Vec::new();
        write_number(&mut out, value);
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn numbers_print_canonically() {
        assert_eq!(number_text(0.0), "0");
        assert_eq!(number_text(-3.0), "-3");
        assert_eq!(number_text(9007199254740992.0), "9007199254740992");
        assert_eq!(number_text(9007199254740994.0), "9.007199254740994e15");
        assert_eq!(number_text(1e21), "1e21");
        assert_eq!(number_text(0.5), "0.5");
        assert_eq!(number_text(0.1), "0.1");
        assert_eq!(number_text(1e-7), "1e-7");
        assert_eq!(number_text(-2.5e300), "-2.5e300");
    }

    #[test]
    fn strings_escape_only_what_json_requires() {
        let mut out = Vec::new();
        write_string(&mut out, "a\"b\\c\n\t\u{1}é😀/");
        assert_eq!(String::from_utf8(out).unwrap(), "\"a\\\"b\\\\c\\n\\t\\u0001é😀/\"");
    }

    #[test]
    fn layout_is_widest_first_and_padded() {
        let mut doc: Map<Tensor> = Map::new();
        doc.insert("bytes".into(), Tensor::from_vec(vec![1u8, 2, 3]).into());
        doc.insert("floats".into(), Tensor::from_vec(vec![1.0f64]).into());
        doc.insert("shorts".into(), Tensor::from_vec(vec![1u16, 2]).into());
        let layout = layout(&doc).unwrap();
        assert_eq!(layout.head.len() % 8, 0);
        let widths: Vec<u64> = layout.tensors.iter().map(|t| t.dtype.width()).collect();
        assert_eq!(widths, vec![8, 2, 1]);
        let json = std::str::from_utf8(&layout.head[8..]).unwrap();
        assert!(json.contains(r#""floats":{"$dtype":"F64","shape":[1],"offset":0,"length":8}"#));
        assert!(json.contains(r#""shorts":{"$dtype":"U16","shape":[2],"offset":8,"length":4}"#));
        assert!(json.contains(r#""bytes":{"$dtype":"U8","shape":[3],"offset":12,"length":3}"#));
        assert_eq!(layout.buffer_length, 15);
    }

    #[test]
    fn quoting_doubles_leading_dollars() {
        let mut doc: Map<Tensor> = Map::new();
        doc.insert("$dtype".into(), Value::Number(1.0));
        doc.insert("$$x".into(), Value::Null);
        doc.insert("a$b".into(), Value::Bool(true));
        let layout = layout(&doc).unwrap();
        let json = std::str::from_utf8(&layout.head[8..]).unwrap().trim_end();
        assert_eq!(json, r#"{"$$dtype":1,"$$$x":null,"a$b":true}"#);
    }

    #[test]
    fn non_finite_numbers_are_refused() {
        let mut doc: Map<Tensor> = Map::new();
        doc.insert("n".into(), Value::Number(f64::NAN));
        assert!(matches!(layout(&doc), Err(Error::NonFinite { .. })));
        let _ = Dtype::F32;
    }
}
