"""Decoding: a document whose tensors are read-only NumPy views over the file's bytes."""

import mmap
import os
from typing import Any

import numpy as np

from .dtypes import DTYPES
from .errors import NeedMore
from .header import Header, TensorRef, declared_length, parse_header, parse_json


def view(buffer: memoryview | bytes, ref: TensorRef) -> np.ndarray:
    """The reference's array, a read-only view over the buffer bytes."""
    dtype = DTYPES[ref.dtype]
    if ref.length == 0:
        return np.frombuffer(b"", dtype=dtype).reshape(ref.shape)
    arr = np.frombuffer(buffer, dtype=dtype, count=ref.elements, offset=ref.offset).reshape(ref.shape)
    arr.flags.writeable = False
    return arr


def decode(data: bytes | bytearray | memoryview | mmap.mmap) -> dict[str, Any]:
    """Decode a file held in memory. Each tensor is a read-only view over the given bytes, not a copy."""
    total = len(data)
    header_length = declared_length(bytes(data[:8]), total)
    end = 8 + header_length
    buffer = memoryview(data)[end:]
    return parse_json(bytes(data[8:end]), total - end, lambda ref: view(buffer, ref))


def read(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a file, mapping it into memory so each tensor is a view and no tensor byte is touched until read.

    The mapping lives as long as any array view over it does.
    """
    with open(path, "rb") as file:
        total = os.fstat(file.fileno()).st_size
        header_length = declared_length(file.read(8), total)
        end = 8 + header_length
        blob = file.read(header_length)
        if total > end:
            mapped = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            buffer: memoryview | bytes = memoryview(mapped)[end:]
        else:
            buffer = b""
    return parse_json(blob, total - end, lambda ref: view(buffer, ref))


def read_header(path: str | os.PathLike[str]) -> Header:
    """Read a file's header alone, touching no tensor bytes."""
    with open(path, "rb") as file:
        total = os.fstat(file.fileno()).st_size
        prefix = file.read(8)
        try:
            return parse_header(prefix, total)
        except NeedMore as need:
            prefix += file.read(need.required - len(prefix))
            return parse_header(prefix, total)
