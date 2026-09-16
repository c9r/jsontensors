# jsontensors

jsontensors keeps a JSON document's arrays out of the text and in a binary buffer behind it, with each array in its place in the document and no copy on read. This package is the Python implementation of the format specified in the [jsontensors repository](https://github.com/c9r/jsontensors).

## Use

```python
import numpy as np
import jsontensors

document = {
    "title": "Rain on a tin roof",
    "samples": np.array([0.25, -0.5, 1.0], dtype=np.float32),
}
jsontensors.write("rain.jsontensors", document)

loaded = jsontensors.read("rain.jsontensors")
loaded["samples"]  # a read-only float32 view over the mapped file
```

A document is a `dict` of JSON values, with NumPy arrays standing where the tensors belong. Any array whose dtype is in the table encodes, whatever its byte order or layout, and decodes as a read-only little-endian view. `bfloat16` arrays are the `ml_dtypes.bfloat16` dtype.

The four operations:

- `read(path)` maps a file and returns the document with tensors as views. The mapping lives as long as any view over it does.
- `read_header(path)` returns a `Header` whose document has a `TensorRef` in each array's place, and touches no tensor bytes.
- `open_ranged(path)` returns a `Ranged` that parsed the header once and serves any byte range of any tensor with one positioned read, which is how to slice a large tensor on a network mount without mapping the file.
- `write(path, document)` streams the document to a temporary file beside the target and renames it into place.

`encode(document)` and `decode(data)` do the same in memory. `parse_header(prefix, total)` takes a prefix of a file and its total size, and raises `NeedMore` naming the bytes it wants when the prefix ends inside the header, which is how a reader over an object store gets the header in two ranged requests.

Every decoding error is a `JsontensorsError` whose `category` names the conformance category, so a caller can tell a truncated file from a malformed one.

## Conformance

`uv run pytest` runs the repository's conformance suite alongside the unit tests.

## License

MIT.
