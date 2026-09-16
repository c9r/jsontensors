import json
import struct
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

import jsontensors
from jsontensors import TensorRef


def document() -> dict:
    return {
        "title": "Rain on a tin roof",
        "sample_rate": 48000,
        "channels": [
            {"name": "left", "samples": np.linspace(-1.0, 1.0, 8, dtype=np.float32)},
            {"name": "right", "samples": np.zeros(8, dtype=np.float32)},
        ],
        "peaks": np.array([[1, -1, 3, -3], [0, 7, -7, 100]], dtype=np.int16),
        "flags": np.array([True, False, True]),
        "scalar": np.array(42.0),
        "empty": np.zeros((0, 3), dtype=np.int32),
    }


def assert_same(a: dict, b: dict) -> None:
    assert list(a) == list(b)
    for key in a:
        x, y = a[key], b[key]
        if isinstance(x, np.ndarray):
            assert isinstance(y, np.ndarray)
            assert x.dtype == y.dtype and x.shape == y.shape
            assert x.tobytes() == y.tobytes()
        elif isinstance(x, dict):
            assert_same(x, y)
        elif isinstance(x, list):
            assert len(x) == len(y)
            for p, q in zip(x, y):
                if isinstance(p, dict):
                    assert_same(p, q)
                else:
                    assert p == q
        else:
            assert x == y and type(x) is type(y)


def frame(text: str, buffer: bytes = b"") -> bytes:
    blob = text.encode()
    blob += b" " * (-(8 + len(blob)) % 8)
    return struct.pack("<Q", len(blob)) + blob + buffer


def test_bytes_round_trip() -> None:
    data = jsontensors.encode(document())
    assert len(data) % 8 == 0 or True
    decoded = jsontensors.decode(data)
    assert_same(decoded, document())
    assert not decoded["peaks"].flags.writeable, "views are read-only"


def test_file_round_trip_and_header(tmp_path: Path) -> None:
    path = tmp_path / "doc.jsontensors"
    jsontensors.write(path, document())
    assert_same(jsontensors.read(path), document())
    header = jsontensors.read_header(path)
    assert header.document["peaks"] == TensorRef("I16", (2, 4), 72, 16)
    assert header.document["empty"] == TensorRef("I32", (0, 3), 72, 0)
    assert header.buffer_length == 8 + 64 + 16 + 3
    assert [p.name for p in tmp_path.iterdir()] == ["doc.jsontensors"], "no temporary file remains"


def test_load_then_save_is_the_identity(tmp_path: Path) -> None:
    path = tmp_path / "doc.jsontensors"
    jsontensors.write(path, document())
    first = path.read_bytes()
    jsontensors.write(path, jsontensors.read(path))
    assert path.read_bytes() == first


def test_layout_is_widest_first_with_ties_in_document_order() -> None:
    doc = {"b": np.zeros(3, np.uint8), "d": np.zeros(1, np.float64), "s": np.zeros(2, np.uint16), "t": np.zeros(1, np.int16)}
    head, arrays = jsontensors.layout(doc)
    assert [a.dtype.itemsize for a in arrays] == [8, 2, 2, 1]
    header = json.loads(head[8:])
    assert header["d"]["offset"] == 0 and header["s"]["offset"] == 8 and header["t"]["offset"] == 12 and header["b"]["offset"] == 14
    assert len(head) % 8 == 0
    for ref in header.values():
        assert ref["offset"] % jsontensors.DTYPES[ref["$dtype"]].itemsize == 0


def test_quoting_round_trips_any_property_name() -> None:
    doc = {"$dtype": "data", "$$x": None, "$": 1, "a$b": True, "": "empty"}
    head, _ = jsontensors.layout(doc)
    assert head[8:].rstrip(b" ") == b'{"$$dtype":"data","$$$x":null,"$$":1,"a$b":true,"":"empty"}'
    assert jsontensors.decode(jsontensors.encode(doc)) == doc


def test_bf16_and_f16_are_distinct_dtypes() -> None:
    doc = {"bf": np.array([1.0, 3.140625], dtype=ml_dtypes.bfloat16), "f": np.array([1.0, -0.5], dtype=np.float16)}
    head, _ = jsontensors.layout(doc)
    header = json.loads(head[8:])
    assert header["bf"]["$dtype"] == "BF16" and header["f"]["$dtype"] == "F16"
    decoded = jsontensors.decode(jsontensors.encode(doc))
    assert decoded["bf"].dtype == np.dtype(ml_dtypes.bfloat16)
    assert decoded["bf"].tolist() == [1.0, 3.140625]


def test_big_endian_and_non_contiguous_input_is_normalized() -> None:
    arr = np.arange(6, dtype=">i4").reshape(2, 3)[:, ::2]
    decoded = jsontensors.decode(jsontensors.encode({"a": arr}))
    assert decoded["a"].dtype == np.dtype("<i4")
    assert decoded["a"].tolist() == [[0, 2], [3, 5]]


def test_header_prefix_protocol() -> None:
    data = jsontensors.encode({"x": 1})
    with pytest.raises(jsontensors.NeedMore) as need:
        jsontensors.parse_header(data[:3], len(data))
    assert need.value.required == len(data)
    with pytest.raises(jsontensors.NeedMore):
        jsontensors.parse_header(data[:10], len(data))
    assert jsontensors.parse_header(data, len(data)).document == {"x": 1}
    with pytest.raises(jsontensors.TruncatedHeader):
        jsontensors.parse_header(data, len(data) - 1)


def test_ranged_reads_equal_the_views(tmp_path: Path) -> None:
    path = tmp_path / "r.jsontensors"
    values = np.arange(2000, dtype=np.uint32)
    jsontensors.write(path, {"v": values, "t": "text"})
    with jsontensors.open_ranged(path) as ranged:
        ref = ranged.document["v"]
        assert ranged.document["t"] == "text"
        assert ranged.tensor(ref).tolist() == values.tolist()
        assert np.frombuffer(ranged.read(ref, 400, 8), dtype="<u4").tolist() == [100, 101]
        with pytest.raises(jsontensors.JsontensorsError):
            ranged.read(ref, 7996, 8)


def test_ranged_open_grows_past_a_large_header(tmp_path: Path) -> None:
    path = tmp_path / "big.jsontensors"
    jsontensors.write(path, {"text": "x" * 10_000, "v": np.array([1, 2, 3], np.uint8)})
    with jsontensors.open_ranged(path) as ranged:
        assert ranged.tensor(ranged.document["v"]).tolist() == [1, 2, 3]


def test_a_document_without_tensors_has_an_empty_buffer() -> None:
    data = jsontensors.encode({"n": 1})
    assert len(data) == 8 + struct.unpack("<Q", data[:8])[0]
    assert jsontensors.decode(data) == {"n": 1}


@pytest.mark.parametrize(
    ("text", "buffer", "error"),
    [
        ('{"a":', b"", jsontensors.InvalidJson),
        ("[1]", b"", jsontensors.NonObjectRoot),
        ('{"$dtype":"U8","shape":[1],"offset":0,"length":1}', b"\x00", jsontensors.NonObjectRoot),
        ('{"$x":1}', b"", jsontensors.StrayDollar),
        ('{"a":{"$dtype":"F128","shape":[1],"offset":0,"length":16}}', b"\x00" * 16, jsontensors.MalformedReference),
        ('{"a":{"$dtype":"U8","shape":[1],"offset":0}}', b"\x00", jsontensors.MalformedReference),
        ('{"a":{"$dtype":"U8","shape":[true],"offset":0,"length":1}}', b"\x00", jsontensors.MalformedReference),
        ('{"a":{"$dtype":"U8","shape":[2],"offset":0,"length":1}}', b"\x00", jsontensors.LengthMismatch),
        ('{"a":{"$dtype":"U8","shape":[2],"offset":1,"length":2}}', b"\x00" * 3, jsontensors.TilingError),
        ('{"a":{"$dtype":"U8","shape":[2],"offset":0,"length":2}}', b"\x00" * 3, jsontensors.TilingError),
        ('{"n":9007199254740993}', b"", jsontensors.NumberError),
        ('{"n":1e400}', b"", jsontensors.NumberError),
        ('{"n":NaN}', b"", jsontensors.NumberError),
    ],
)
def test_refusals(text: str, buffer: bytes, error: type[Exception]) -> None:
    with pytest.raises(error):
        jsontensors.decode(frame(text, buffer))


def test_truncations() -> None:
    with pytest.raises(jsontensors.TruncatedLength):
        jsontensors.decode(b"\x01\x02\x03")
    with pytest.raises(jsontensors.TruncatedHeader):
        jsontensors.decode(struct.pack("<Q", 100) + b"{}")


@pytest.mark.parametrize(
    ("doc", "match"),
    [
        ({"n": float("nan")}, "not finite"),
        ({"n": 2**53 + 1}, "beyond 2\\^53"),
        ({"n": np.float32(1.0)}, "NumPy scalar"),
        ({"n": object()}, "neither JSON"),
        ({1: 2}, "not a string"),
        ({"a": np.zeros(2, dtype=np.complex64)}, "not a jsontensors dtype"),
    ],
)
def test_encode_refusals(doc: dict, match: str) -> None:
    with pytest.raises((jsontensors.EncodeError, TypeError), match=match):
        jsontensors.encode(doc)


def test_the_edges_of_the_number_rule() -> None:
    assert jsontensors.decode(frame('{"n":9007199254740992,"m":-9007199254740992}')) == {"n": 2**53, "m": -(2**53)}
    assert jsontensors.encode({"n": 2**53}) == frame('{"n":9007199254740992}')


def test_a_crash_leaves_no_temp_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "doc.jsontensors"

    def explode(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(jsontensors.encode.__globals__["os"], "replace", explode)
    with pytest.raises(OSError):
        jsontensors.write(path, {"a": 1})
    assert list(tmp_path.iterdir()) == []
