# jsontensors

jsontensors keeps a JSON document's arrays out of the text and in a binary buffer behind it, with each array in its place in the document and no copy on read. This package is the TypeScript implementation of the format specified in the [jsontensors repository](https://github.com/c9r/jsontensors).

## Use

```ts
import { Tensor, decode, encode } from "jsontensors";
import { read, write } from "jsontensors/node";

const document = {
  title: "Rain on a tin roof",
  samples: Tensor.from(Float32Array.from([0.25, -0.5, 1])),
};
write("rain.jsontensors", document);

const loaded = read("rain.jsontensors");
const samples = loaded.samples as Tensor;
samples.array(); // a Float32Array view over the file's bytes
```

A document is a plain object of JSON values, with `Tensor` instances standing where the arrays belong. `Tensor.from` wraps a typed array, taking its dtype from the array's type, or the dtype you name for `BF16` over a `Uint16Array` and `BOOL` over a `Uint8Array`. `tensor.array()` is the dtype's typed array, a view when the bytes are aligned and a copy when they are not. `BF16` decodes as its `Uint16Array` bit patterns, and `bf16ToFloat32` converts them.

The `jsontensors` entry point is runtime-neutral and works wherever `Uint8Array` does: `decode(bytes)` and `encode(document)`, `parseHeader(prefix, total)` for a reader over an object store that fetches the header in two ranged requests, and `chunks(document)` for a writer that streams. The `jsontensors/node` entry point adds the file operations:

- `read(path)` reads a file and returns the document with tensors as views over its bytes.
- `readHeader(path)` returns the document with a `TensorRef` in each array's place, and touches no tensor bytes.
- `openRanged(path)` returns a `Ranged` that parsed the header once and serves any byte range of any tensor with one positioned read.
- `write(path, document)` writes to a temporary file beside the target and renames it into place.

Every decoding error is a `JsontensorsError` whose `category` names the conformance category, so a caller can tell a truncated file from a malformed one.

## Conformance

`pnpm test` runs the repository's conformance suite alongside the unit tests.

## License

MIT.
