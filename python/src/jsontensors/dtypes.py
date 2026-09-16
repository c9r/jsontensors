"""The dtype table: the element types a tensor can hold, as NumPy dtypes."""

import ml_dtypes
import numpy as np

BF16 = np.dtype(ml_dtypes.bfloat16)

# Every dtype name to the little-endian NumPy dtype a tensor of it decodes as.
DTYPES: dict[str, np.dtype] = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "BF16": BF16,
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("i1"),
    "U64": np.dtype("<u8"),
    "U32": np.dtype("<u4"),
    "U16": np.dtype("<u2"),
    "U8": np.dtype("u1"),
    "BOOL": np.dtype("?"),
}


def width(name: str) -> int:
    """The width of one element of a dtype, in bytes."""
    return DTYPES[name].itemsize


def name_of(dtype: np.dtype) -> str:
    """The dtype name of a NumPy dtype, whatever its byte order.

    Raises ``TypeError`` for a dtype the table does not hold.
    """
    if dtype == BF16:
        return "BF16"
    for name, target in DTYPES.items():
        if name != "BF16" and dtype.kind == target.kind and dtype.itemsize == target.itemsize:
            return name
    raise TypeError(f"{dtype} is not a jsontensors dtype")
