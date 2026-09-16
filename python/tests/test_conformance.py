"""Runs every case in conformance/cases against this implementation."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import jsontensors

CASES = Path(__file__).resolve().parent.parent.parent / "conformance" / "cases"
STEMS = sorted(path.stem for path in CASES.glob("*.jsontensors"))


def materialize(document: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    """The decoded document in the expectation's form: ``None`` at each tensor, and the tensors listed by pointer."""
    tensors: list[dict[str, Any]] = []

    def walk(value: Any, path: str) -> Any:
        if isinstance(value, np.ndarray):
            data = np.ascontiguousarray(value).tobytes()
            tensors.append(
                {
                    "path": path,
                    "dtype": jsontensors.name_of(value.dtype),
                    "shape": list(value.shape),
                    "length": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            return None
        if isinstance(value, dict):
            return {k: walk(v, path + "/" + k.replace("~", "~0").replace("/", "~1")) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v, f"{path}/{i}") for i, v in enumerate(value)]
        return value

    return walk(document, ""), tensors


def same(a: Any, b: Any) -> bool:
    """Equality of JSON values, where bools are not numbers and every number is a double."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=0, abs_tol=0)
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


@pytest.mark.parametrize("stem", STEMS)
def test_case(stem: str) -> None:
    data = (CASES / f"{stem}.jsontensors").read_bytes()
    expected_path = CASES / f"{stem}.expected.json"
    error_path = CASES / f"{stem}.error.json"
    if expected_path.exists():
        expected = json.loads(expected_path.read_text())
        decoded = jsontensors.decode(data)
        document, tensors = materialize(decoded)
        assert same(document, expected["document"]), f"{stem}: decoded document differs"
        assert same(tensors, expected["tensors"]), f"{stem}: tensors differ"
        re_encoded = jsontensors.encode(decoded)
        if expected["canonical"]:
            assert re_encoded == data, f"{stem}: a canonical case did not reproduce byte for byte"
        else:
            assert same(materialize(jsontensors.decode(re_encoded)), (document, tensors)), f"{stem}: re-encoding changed values"
    elif error_path.exists():
        accepted = json.loads(error_path.read_text())["error"]
        accepted = [accepted] if isinstance(accepted, str) else accepted
        with pytest.raises(jsontensors.JsontensorsError) as caught:
            jsontensors.decode(data)
        assert caught.value.category in accepted, f"{stem}: refused as {caught.value.category}, expected {accepted}"
    else:
        pytest.fail(f"{stem} has neither an expectation nor an error file")


def test_cases_exist() -> None:
    assert STEMS, f"no cases in {CASES}; run `cargo run --example generate` in rust/"
