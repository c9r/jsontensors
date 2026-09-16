//! Decoding: a document whose tensors are views over the file's bytes.

use std::fs::File;
use std::path::Path;

use bytes::Bytes;

use crate::error::Error;
use crate::header::{Header, parse_header};
use crate::value::{Document, Tensor, TensorRef, map_tensors_in};

/// Decodes a file held in memory. Each tensor's bytes are a slice of the given bytes, not a copy.
pub fn decode(data: impl Into<Bytes>) -> Result<Document<Tensor>, Error> {
    let data: Bytes = data.into();
    let header = parse_header(&data, data.len() as u64)?;
    resolve(header, &data)
}

/// Replaces each reference in a parsed header with a tensor over the file's bytes.
pub fn resolve(header: Header, file: &Bytes) -> Result<Document<Tensor>, Error> {
    let start = header.buffer_start as usize;
    map_tensors_in(header.document, &mut |reference: TensorRef| {
        let begin = start + reference.offset as usize;
        let end = begin + reference.length as usize;
        Ok(Tensor { dtype: reference.dtype, shape: reference.shape, data: file.slice(begin..end) })
    })
}

/// Reads a file, mapping it into memory so each tensor is a view and no tensor byte is touched until read.
pub fn read(path: impl AsRef<Path>) -> Result<Document<Tensor>, Error> {
    let file = File::open(path)?;
    let mapped = unsafe { memmap2::Mmap::map(&file)? };
    decode(Bytes::from_owner(mapped))
}

/// Reads a file's header alone, touching no tensor bytes.
pub fn read_header(path: impl AsRef<Path>) -> Result<Header, Error> {
    use std::io::Read;
    let mut file = File::open(path)?;
    let total = file.metadata()?.len();
    let mut prefix = vec![0u8; 8.min(total as usize)];
    file.read_exact(&mut prefix)?;
    match parse_header(&prefix, total) {
        Err(Error::NeedMore { required }) => {
            prefix.resize(required as usize, 0);
            file.read_exact(&mut prefix[8..])?;
            parse_header(&prefix, total)
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{encode, write};
    use crate::value::{Map, Value};

    fn document() -> Document<Tensor> {
        let mut doc: Map<Tensor> = Map::new();
        doc.insert("name".into(), "rain".into());
        doc.insert("samples".into(), Tensor::from_vec(vec![0.5f32, -1.5, 2.0]).into());
        let mut inner: Map<Tensor> = Map::new();
        inner.insert("flags".into(), crate::value::bools(vec![2], &[true, false]).unwrap().into());
        doc.insert("inner".into(), Value::Array(vec![Value::Object(inner)]));
        doc
    }

    #[test]
    fn decode_views_the_bytes() {
        let bytes = encode(&document()).unwrap();
        let decoded = decode(bytes.clone()).unwrap();
        assert_eq!(decoded, document());
        let Value::Tensor(samples) = &decoded["samples"] else { panic!() };
        assert_eq!(samples.elements_of::<f32>().unwrap().as_ref(), &[0.5, -1.5, 2.0]);
    }

    #[test]
    fn read_maps_the_file_and_read_header_touches_no_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("doc.jsontensors");
        write(&path, &document()).unwrap();
        assert_eq!(read(&path).unwrap(), document());
        let header = read_header(&path).unwrap();
        let Value::Tensor(reference) = &header.document["samples"] else { panic!() };
        assert_eq!(reference.length, 12);
        assert_eq!(header.buffer_length, 14);
        assert!(dir.path().read_dir().unwrap().count() == 1, "no temporary file remains");
    }

    #[test]
    fn a_document_without_tensors_has_an_empty_buffer() {
        let mut doc: Map<Tensor> = Map::new();
        doc.insert("n".into(), Value::Number(1.0));
        let bytes = encode(&doc).unwrap();
        assert_eq!(bytes.len() % 8, 0);
        assert_eq!(decode(bytes).unwrap(), doc);
    }
}
