//! Ranged reads: the header parsed once, the tensors left where they are, and
//! any byte range of any tensor read on request with one positioned read.
//!
//! This is how a consumer slices a large tensor on a network-backed mount,
//! where faulting a mapping would fetch the slice one page at a time.

use std::fs::File;
use std::path::Path;

use bytes::Bytes;

use crate::error::Error;
use crate::header::{Header, parse_header};
use crate::value::{Tensor, TensorRef};

/// Something a header and byte ranges can be read from at absolute offsets.
pub trait Source {
    /// The total byte count.
    fn size(&self) -> Result<u64, Error>;
    /// Fills `buf` from the bytes starting at `offset`, failing if they are not all there.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), Error>;
}

impl Source for File {
    fn size(&self) -> Result<u64, Error> {
        Ok(self.metadata()?.len())
    }

    #[cfg(unix)]
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), Error> {
        use std::os::unix::fs::FileExt;
        Ok(self.read_exact_at(buf, offset)?)
    }

    #[cfg(windows)]
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), Error> {
        use std::os::windows::fs::FileExt;
        let mut done = 0;
        while done < buf.len() {
            let n = self.seek_read(&mut buf[done..], offset + done as u64)?;
            if n == 0 {
                return Err(
                    std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "the file ended inside a range").into()
                );
            }
            done += n;
        }
        Ok(())
    }
}

/// A source opened for ranged reads, with its header parsed.
pub struct Ranged<S: Source> {
    source: S,
    header: Header,
}

impl<S: Source> Ranged<S> {
    /// Parses the source's header, reading a small prefix and then exactly what the header needs.
    pub fn open(source: S) -> Result<Ranged<S>, Error> {
        let total = source.size()?;
        let mut prefix = vec![0u8; 4096.min(total) as usize];
        source.read_at(0, &mut prefix)?;
        let header = match parse_header(&prefix, total) {
            Err(Error::NeedMore { required }) => {
                let have = prefix.len();
                prefix.resize(required as usize, 0);
                source.read_at(have as u64, &mut prefix[have..])?;
                parse_header(&prefix, total)?
            }
            other => other?,
        };
        Ok(Ranged { source, header })
    }

    /// The header: the document with references in the arrays' places.
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// The underlying source.
    pub fn source(&self) -> &S {
        &self.source
    }

    /// `length` bytes of a reference starting `start` bytes into it.
    pub fn read(&self, reference: &TensorRef, start: u64, length: u64) -> Result<Bytes, Error> {
        let end = start.checked_add(length).filter(|&end| end <= reference.length);
        let Some(end) = end else {
            return Err(Error::RangeOutside { start, end: start.saturating_add(length), length: reference.length });
        };
        let mut buf = vec![0u8; length as usize];
        self.source.read_at(self.header.buffer_start + reference.offset + start, &mut buf)?;
        let _ = end;
        Ok(Bytes::from(buf))
    }

    /// A reference's whole tensor.
    pub fn tensor(&self, reference: &TensorRef) -> Result<Tensor, Error> {
        let data = self.read(reference, 0, reference.length)?;
        Ok(Tensor { dtype: reference.dtype, shape: reference.shape.clone(), data })
    }
}

/// Opens a file for ranged reads.
pub fn open(path: impl AsRef<Path>) -> Result<Ranged<File>, Error> {
    Ranged::open(File::open(path)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::write;
    use crate::value::{Map, Value};

    #[test]
    fn ranged_reads_match_the_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("r.jsontensors");
        let mut doc: Map<Tensor> = Map::new();
        let values: Vec<u32> = (0..2000).collect();
        doc.insert("v".into(), Tensor::from_vec(values.clone()).into());
        write(&path, &doc).unwrap();
        let ranged = open(&path).unwrap();
        let Value::Tensor(reference) = &ranged.header().document["v"] else { panic!() };
        let whole = ranged.tensor(reference).unwrap();
        assert_eq!(whole.to_vec::<u32>().unwrap(), values);
        let slice = ranged.read(reference, 400, 8).unwrap();
        assert_eq!(bytemuck::pod_collect_to_vec::<u8, u32>(&slice), vec![100, 101]);
        assert!(ranged.read(reference, 7996, 8).is_err());
    }

    #[test]
    fn open_grows_the_prefix_past_a_large_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.jsontensors");
        let mut doc: Map<Tensor> = Map::new();
        doc.insert("text".into(), Value::String("x".repeat(10_000)));
        doc.insert("v".into(), Tensor::from_vec(vec![1u8, 2, 3]).into());
        write(&path, &doc).unwrap();
        let ranged = open(&path).unwrap();
        let Value::Tensor(reference) = &ranged.header().document["v"] else { panic!() };
        assert_eq!(ranged.tensor(reference).unwrap().data.as_ref(), &[1, 2, 3]);
    }
}
