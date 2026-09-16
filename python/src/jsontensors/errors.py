"""Errors, each carrying the conformance category it belongs to."""


class JsontensorsError(ValueError):
    """Any error the format raises. ``category`` names the conformance category of a decoding error, or is ``None``."""

    category: str | None = None


class TruncatedLength(JsontensorsError):
    category = "truncated-length"


class TruncatedHeader(JsontensorsError):
    category = "truncated-header"


class InvalidJson(JsontensorsError):
    category = "invalid-json"


class NonObjectRoot(JsontensorsError):
    category = "non-object-root"


class MalformedReference(JsontensorsError):
    category = "malformed-reference"


class LengthMismatch(JsontensorsError):
    category = "length-mismatch"


class StrayDollar(JsontensorsError):
    category = "stray-dollar"


class TilingError(JsontensorsError):
    category = "tiling"


class NumberError(JsontensorsError):
    category = "number"


class NeedMore(JsontensorsError):
    """A prefix ended inside the header of a file that does carry it whole. Retry with at least ``required`` bytes."""

    def __init__(self, required: int) -> None:
        super().__init__(f"the header needs {required} bytes")
        self.required = required


class EncodeError(JsontensorsError):
    """A document the encoder refuses: a number JSON cannot carry exactly, or a value that is neither JSON nor an array."""
