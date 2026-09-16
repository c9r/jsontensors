"""Ranged reads: the header parsed once, the tensors left where they are, and any byte range of any tensor read on request.

This is how a consumer slices a large tensor on a network-backed mount, where
faulting a mapping would fetch the slice one page at a time.
"""

import os
from types import TracebackType
from typing import Any

import numpy as np

from .dtypes import DTYPES
from .errors import JsontensorsError, NeedMore
from .header import FIRST_PREFIX, Header, TensorRef, parse_header


class Ranged:
    """A file opened for ranged reads, with its header parsed."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = os.fspath(path)
        self.file = open(self.path, "rb")
        try:
            total = os.fstat(self.file.fileno()).st_size
            prefix = self.file.read(min(total, FIRST_PREFIX))
            try:
                self.header: Header = parse_header(prefix, total)
            except NeedMore as need:
                prefix += self.file.read(need.required - len(prefix))
                self.header = parse_header(prefix, total)
        except BaseException:
            self.file.close()
            raise

    @property
    def document(self) -> dict[str, Any]:
        """The document with references in the arrays' places."""
        return self.header.document

    def read(self, ref: TensorRef, start: int, length: int) -> bytes:
        """``length`` bytes of a reference starting ``start`` bytes into it, with one positioned read."""
        if start < 0 or length < 0 or start + length > ref.length:
            raise JsontensorsError(f"the range [{start}, {start + length}) lies outside a {ref.length}-byte reference")
        return os.pread(self.file.fileno(), length, self.header.buffer_start + ref.offset + start)

    def tensor(self, ref: TensorRef) -> np.ndarray:
        """A reference's whole tensor, read into memory."""
        data = self.read(ref, 0, ref.length)
        return np.frombuffer(data, dtype=DTYPES[ref.dtype]).reshape(ref.shape)

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "Ranged":
        return self

    def __exit__(self, kind: type[BaseException] | None, value: BaseException | None, tb: TracebackType | None) -> None:
        self.close()


def open_ranged(path: str | os.PathLike[str]) -> Ranged:
    """Open a file for ranged reads."""
    return Ranged(path)
