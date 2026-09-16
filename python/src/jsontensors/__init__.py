"""jsontensors keeps a JSON document's arrays out of the text and in a binary buffer behind it, with each array in its place in the document and no copy on read.

A file is an eight-byte little-endian header length, that many bytes of UTF-8
JSON padded to an eight-byte boundary, and then the tensors' bytes. In the
JSON, each array's place holds a reference naming its dtype, shape, and byte
range. This package transcribes the specification in the repository's SPEC.md.

The four operations:

- ``read`` maps a file and returns the document with each tensor a read-only NumPy view over the mapping.
- ``read_header`` returns the document with a ``TensorRef`` in each array's place, touching no tensor bytes.
- ``Ranged`` parses the header once and reads any byte range of any tensor with one positioned read.
- ``write`` streams a document to a temporary file beside the target and renames it into place.

``encode`` and ``decode`` do the same in memory, for documents that travel as bytes.
"""

from .decode import decode, read, read_header
from .dtypes import DTYPES, name_of
from .encode import encode, layout, write, write_to
from .errors import (
    EncodeError,
    InvalidJson,
    JsontensorsError,
    LengthMismatch,
    MalformedReference,
    NeedMore,
    NonObjectRoot,
    NumberError,
    StrayDollar,
    TilingError,
    TruncatedHeader,
    TruncatedLength,
)
from .header import Header, TensorRef, parse_header
from .ranged import Ranged, open_ranged

SUFFIX = ".jsontensors"
MEDIA_TYPE = "application/x-jsontensors"

__all__ = [
    "DTYPES",
    "EncodeError",
    "Header",
    "InvalidJson",
    "JsontensorsError",
    "LengthMismatch",
    "MEDIA_TYPE",
    "MalformedReference",
    "NeedMore",
    "NonObjectRoot",
    "NumberError",
    "Ranged",
    "SUFFIX",
    "StrayDollar",
    "TensorRef",
    "TilingError",
    "TruncatedHeader",
    "TruncatedLength",
    "decode",
    "encode",
    "layout",
    "name_of",
    "open_ranged",
    "parse_header",
    "read",
    "read_header",
    "write",
    "write_to",
]
