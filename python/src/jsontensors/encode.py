"""Encoding: laying tensors out, serializing the header canonically, and writing the file."""

import json
import math
import os
import struct
import tempfile
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from .dtypes import DTYPES, name_of
from .errors import EncodeError
from .header import MAX_SAFE_INTEGER, pointer


def layout(document: dict[str, Any]) -> tuple[bytes, list[np.ndarray]]:
    """The length-prefixed padded header and the arrays in buffer order.

    Arrays are collected in document order and laid out widest dtype first,
    ties in collection order, so with the buffer starting on an 8-byte
    boundary every tensor begins at a multiple of its width. Identical input
    produces identical bytes.
    """
    if not isinstance(document, dict):
        raise EncodeError("the document is not a JSON object")
    arrays: list[np.ndarray] = []
    references: list[dict[str, Any]] = []
    transformed = encoded(document, [], arrays, references)
    order = sorted(range(len(arrays)), key=lambda i: -arrays[i].dtype.itemsize)
    offset = 0
    for i in order:
        arr = arrays[i]
        references[i]["$dtype"] = name_of(arr.dtype)
        references[i]["shape"] = list(arr.shape)
        references[i]["offset"] = offset
        references[i]["length"] = arr.nbytes
        offset += arr.nbytes
    blob = json.dumps(transformed, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    blob += b" " * (-(8 + len(blob)) % 8)
    return struct.pack("<Q", len(blob)) + blob, [arrays[i] for i in order]


def encoded(value: Any, path: list[str | int], arrays: list[np.ndarray], references: list[dict[str, Any]]) -> Any:
    """The value's header form, with arrays collected and their references planted empty, to be filled once the layout is known."""
    if isinstance(value, np.ndarray):
        name = name_of(value.dtype)
        arr = np.asarray(value, dtype=DTYPES[name], order="C")
        reference: dict[str, Any] = {}
        arrays.append(arr)
        references.append(reference)
        return reference
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise EncodeError(f"the property name {key!r} at {pointer(path)} is not a string")
            path.append(key)
            out["$" + key if key.startswith("$") else key] = encoded(item, path, arrays, references)
            path.pop()
        return out
    if isinstance(value, list):
        out_list = []
        for index, item in enumerate(value):
            path.append(index)
            out_list.append(encoded(item, path, arrays, references))
            path.pop()
        return out_list
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        if abs(value) > MAX_SAFE_INTEGER:
            raise EncodeError(f"the integer at {pointer(path)} is beyond 2^53 and not exactly representable as a double")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EncodeError(f"the number at {pointer(path)} is not finite, and JSON cannot spell it")
        return value
    if isinstance(value, np.generic):
        raise EncodeError(
            f"the value at {pointer(path)} is a NumPy scalar; pass a Python number for JSON or a 0-d array for a tensor"
        )
    raise EncodeError(f"the value at {pointer(path)} is a {type(value).__name__}, which is neither JSON nor an array")


def array_bytes(arr: np.ndarray) -> memoryview:
    """A C-contiguous array's bytes without a copy, whatever its dtype, as a flat view of unsigned bytes."""
    return arr.reshape(-1).view(np.uint8).data


def write_to(file: BinaryIO, document: dict[str, Any]) -> None:
    """Write a document to a binary file object, head first and then each array's bytes in buffer order."""
    head, arrays = layout(document)
    file.write(head)
    for arr in arrays:
        file.write(array_bytes(arr))


def encode(document: dict[str, Any]) -> bytes:
    """The whole file as bytes."""
    head, arrays = layout(document)
    return b"".join([head, *(array_bytes(arr) for arr in arrays)])


def write(path: str | os.PathLike[str], document: dict[str, Any]) -> None:
    """Write a document to a path atomically: a temporary file in the target's directory, synced, then renamed over the target."""
    path = Path(path)
    head, arrays = layout(document)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".jsontensors-")
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(head)
            for arr in arrays:
                file.write(array_bytes(arr))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, path)
    except BaseException:
        if os.path.exists(temp):
            os.unlink(temp)
        raise
