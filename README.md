# jsontensors

jsontensors keeps a JSON document's arrays out of the text and in a binary buffer behind it, with each array in its place in the document and no copy on read.

The document owns the file. It stays JSON, any JSON at all, and reads as JSON. The arrays stay bytes, typed and shaped, and read as views. It is safetensors's container with a document in the header slot and the references where the arrays belong.

## Why

JSON holds structure well and numbers badly. A float costs twenty bytes as text, reading it means parsing it, and a document whose bulk is numeric is mostly parser time and mostly bloat. A tensor file holds numbers well and structure badly. It is a flat table of named arrays. The shape of the data, which array belongs to what, lives in code somewhere else. Most numeric data is a little structure around a lot of numbers. The two families make you choose which half to do badly. jsontensors makes no choice. The structure is JSON and the numbers are bytes, in one file. Each side is read the way it should be.

## The format

A file is an eight-byte little-endian header length, that many bytes of UTF-8 JSON padded with spaces to an eight-byte boundary, and then the tensors' bytes, contiguous. In the JSON, each array's place holds a reference:

```json
{ "$dtype": "F32", "shape": [96000], "offset": 0, "length": 384000 }
```

A decoder parses the JSON and replaces each reference with a view over the buffer. An encoder does the inverse, laying tensors out widest dtype first so every view is aligned. A property name that begins with `$` travels with one more `$`, so `$dtype` can only ever be written by the format. Any JSON document therefore round-trips unchanged. The references must tile the buffer exactly, so a truncated or spliced file is refused rather than misread. [SPEC.md](SPEC.md) is the whole specification, and it is short.

## Implementations

Three implementations live here, each a complete package, each passing the same conformance suite.

| Language | Package | Directory | Arrays decode as |
| --- | --- | --- | --- |
| Rust | `jsontensors` on crates.io | `rust/` | typed slices over a mapping or a buffer |
| Python | `jsontensors` on PyPI | `python/` | read-only NumPy views over a mapping or a buffer |
| TypeScript | `jsontensors` on npm | `typescript/` | typed-array views over a buffer |

Each exposes the same four operations with the same semantics:

- **read** a file as a document whose arrays are views over a mapping of it
- **read the header** alone, as the document with references in the arrays' places, touching no tensor bytes
- **read a range** of one reference's bytes from a file or an object store, for a consumer that wants a slice of a large array without mapping the file
- **write** a document, streaming its arrays to a temporary file beside the target and renaming into place

And each encodes and decodes in memory, for documents that travel as bytes.

## Conformance

`conformance/cases` holds files every implementation must decode to the stated values or refuse for the stated reason, and files every implementation must reproduce byte for byte after a decode and re-encode. [conformance/README.md](conformance/README.md) describes the cases. Each implementation's test suite runs them. An implementation that passes is a jsontensors implementation.

## License

MIT. See `LICENSE`.
