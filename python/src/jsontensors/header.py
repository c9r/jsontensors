"""Parsing the header: the length, the JSON, and the references, validated against the buffer."""

import json
import math
import struct
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .dtypes import DTYPES
from .errors import (
    InvalidJson,
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

# The largest integer magnitude a double represents exactly.
MAX_SAFE_INTEGER = 2**53

# The prefix a reader fetches first, before it knows the header's length.
FIRST_PREFIX = 4096

REFERENCE_KEYS = frozenset({"$dtype", "shape", "offset", "length"})


@dataclass(frozen=True)
class TensorRef:
    """A reference: where a tensor's bytes are, without the bytes."""

    dtype: str
    shape: tuple[int, ...]
    # The first byte, counted from the start of the buffer.
    offset: int
    # The byte count, which is the element count times the dtype's width.
    length: int

    @property
    def elements(self) -> int:
        """The element count, which is one for a scalar."""
        return math.prod(self.shape)

    @property
    def end(self) -> int:
        """The byte after the last, counted from the start of the buffer."""
        return self.offset + self.length


@dataclass(frozen=True)
class Header:
    """A parsed header: the document with references in the arrays' places, and where the buffer is."""

    document: dict[str, Any]
    # The header's byte count, padding included, as the length prefix states it.
    header_length: int
    # The buffer's first byte, counted from the start of the file.
    buffer_start: int
    # The buffer's byte count, which the references must tile exactly.
    buffer_length: int


def declared_length(prefix: bytes, total: int) -> int:
    """The header length a prefix declares, checked against the file's total size."""
    if total < 8:
        raise TruncatedLength("the file ends before the eight length bytes")
    if len(prefix) < 8:
        raise NeedMore(min(total, FIRST_PREFIX))
    (length,) = struct.unpack("<Q", prefix[:8])
    if length > total - 8:
        raise TruncatedHeader(f"the header length {length} overruns the {total}-byte file")
    return length


def parse_header(prefix: bytes, total: int) -> Header:
    """Parse the header from a prefix of a file whose total size is known.

    A reader with an object store or a socket learns the size from the store,
    reads a first prefix, and retries with the size a ``NeedMore`` names until
    the header is in hand. The references validate against the total, so a
    truncated object fails here exactly as a truncated file does.
    """
    header_length = declared_length(prefix, total)
    end = 8 + header_length
    if len(prefix) < end:
        raise NeedMore(end)
    buffer_length = total - end
    document = parse_json(bytes(prefix[8:end]), buffer_length, lambda ref: ref)
    return Header(document, header_length, end, buffer_length)


def parse_json(blob: bytes, buffer_length: int, tensor: Callable[[TensorRef], Any]) -> dict[str, Any]:
    """Parse the header JSON, validate its references against the buffer, and substitute each through ``tensor``."""
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError as error:
        raise InvalidJson("the header is not UTF-8") from error
    try:
        raw = json.loads(text, parse_int=parse_int, parse_float=parse_float, parse_constant=parse_constant)
    except json.JSONDecodeError as error:
        raise InvalidJson(f"the header is not JSON: {error}") from error
    if not isinstance(raw, dict):
        raise NonObjectRoot("the header's root is not an object")
    if "$dtype" in raw:
        raise NonObjectRoot("the header's root is a reference, not an object")
    references: list[TensorRef] = []
    document = convert_object(raw, [], references)
    check_tiling(references, buffer_length)
    return substitute(document, tensor)


def substitute(value: Any, tensor: Callable[[TensorRef], Any]) -> Any:
    """The value with every reference replaced through ``tensor``, once the references have been validated."""
    if isinstance(value, TensorRef):
        return tensor(value)
    if isinstance(value, dict):
        return {key: substitute(item, tensor) for key, item in value.items()}
    if isinstance(value, list):
        return [substitute(item, tensor) for item in value]
    return value


def parse_int(literal: str) -> int:
    value = int(literal)
    if abs(value) > MAX_SAFE_INTEGER:
        raise NumberError(f"the integer {literal} is not exactly representable as a double")
    return value


def parse_float(literal: str) -> float:
    value = float(literal)
    if not math.isfinite(value):
        raise NumberError(f"the number {literal} overflows a double")
    return value


def parse_constant(literal: str) -> float:
    raise NumberError(f"the header contains {literal}, which JSON cannot spell")


def pointer(path: list[str | int]) -> str:
    """A path as a JSON Pointer, or ``the root`` for the empty path."""
    if not path:
        return "the root"
    return "".join("/" + (str(p) if isinstance(p, int) else p.replace("~", "~0").replace("/", "~1")) for p in path)


def convert(raw: Any, path: list[str | int], references: list[TensorRef]) -> Any:
    """The value's document form, unquoted, with references parsed, gathered, and left in place."""
    if isinstance(raw, dict):
        if "$dtype" in raw:
            reference = parse_reference(raw, path)
            references.append(reference)
            return reference
        return convert_object(raw, path, references)
    if isinstance(raw, list):
        out = []
        for index, item in enumerate(raw):
            path.append(index)
            out.append(convert(item, path, references))
            path.pop()
        return out
    return raw


def convert_object(raw: dict[str, Any], path: list[str | int], references: list[TensorRef]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, item in raw.items():
        if key.startswith("$$"):
            name = key[1:]
        elif key.startswith("$"):
            raise StrayDollar(f"the property {key!r} at {pointer(path)} has a single leading $ outside a reference")
        else:
            name = key
        path.append(key)
        out[name] = convert(item, path, references)
        path.pop()
    return out


def parse_reference(raw: dict[str, Any], path: list[str | int]) -> TensorRef:
    where = pointer(path)
    if set(raw) != REFERENCE_KEYS:
        raise MalformedReference(f"the reference at {where} is malformed: its properties are {sorted(raw)}")
    dtype = raw["$dtype"]
    shape = raw["shape"]
    offset = raw["offset"]
    length = raw["length"]
    if not isinstance(dtype, str) or dtype not in DTYPES:
        raise MalformedReference(f"the reference at {where} is malformed: {dtype!r} is not a dtype")
    if not isinstance(shape, list) or not all(is_size(n) for n in shape):
        raise MalformedReference(f"the reference at {where} is malformed: shape is not a list of non-negative integers")
    if not is_size(offset):
        raise MalformedReference(f"the reference at {where} is malformed: offset is not a non-negative integer")
    if not is_size(length):
        raise MalformedReference(f"the reference at {where} is malformed: length is not a non-negative integer")
    expected = math.prod(shape) * DTYPES[dtype].itemsize
    if length != expected:
        raise LengthMismatch(
            f"the reference at {where} has length {length}, but its shape and dtype make {expected} bytes"
        )
    return TensorRef(dtype, tuple(shape), offset, length)


def is_size(value: Any) -> bool:
    """Whether a value is a non-negative integer within the safe range, and not a bool."""
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= MAX_SAFE_INTEGER


def check_tiling(references: list[TensorRef], buffer_length: int) -> None:
    """Check that the references, ordered by offset, tile the buffer exactly."""
    cursor = 0
    for begin, end in sorted((r.offset, r.end) for r in references):
        if begin > cursor:
            raise TilingError(f"the references do not tile the buffer: a gap of {begin - cursor} bytes before offset {begin}")
        if begin < cursor:
            raise TilingError(f"the references do not tile the buffer: the reference at offset {begin} overlaps the one before it")
        cursor = end
    if cursor != buffer_length:
        raise TilingError(f"the references do not tile the buffer: they end at {cursor}, but the buffer is {buffer_length} bytes")
