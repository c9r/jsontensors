# jsontensors

jsontensors keeps a JSON document's arrays out of the text and in a binary buffer behind it, with each array in its place in the document and no copy on read. This crate is the Rust implementation of the format specified in the [jsontensors repository](https://github.com/c9r/jsontensors).

## Use

```rust
use jsontensors::{Document, Tensor, Value};

let mut document: Document<Tensor> = Document::new();
document.insert("title".into(), "Rain on a tin roof".into());
document.insert("samples".into(), Tensor::from_vec(vec![0.25f32, -0.5, 1.0]).into());

jsontensors::write("rain.jsontensors".as_ref(), &document)?;

let loaded = jsontensors::read("rain.jsontensors")?;
let Value::Tensor(samples) = &loaded["samples"] else { unreachable!() };
let values: &[f32] = samples.as_slice()?;
```

A document is an ordered map of JSON values, with [`Tensor`] standing where an array belongs. Reading a file maps it, so each tensor is a view over the mapping and no tensor byte is touched until it is read. A tensor's elements come out as a typed slice with `as_slice`, or with `elements_of` when the bytes might not be aligned, or as bools with `to_bools`.

The four operations:

- `read` maps a file and returns the document with tensors as views.
- `read_header` returns the document with a `TensorRef` in each array's place and touches no tensor bytes.
- `ranged::open` parses the header once and serves any byte range of any tensor with one positioned read, which is how to slice a large tensor on a network mount without mapping the file.
- `write` streams the document to a temporary file beside the target and renames it into place.

`encode` and `decode` do the same in memory. `parse_header` takes a prefix of a file and its total size, and asks for more bytes when the prefix ends inside the header, which is how a reader over an object store gets the header in two ranged requests.

## Command line

With the `cli` feature, the crate installs a `jsontensors` binary that prints a file's header, checks it, or lists its tensors:

```sh
cargo install jsontensors --features cli
jsontensors ls rain.jsontensors
```

## Conformance

`cargo test` runs the repository's conformance suite, and `cargo run --example generate` regenerates it. The valid cases are this crate's own output, and every implementation in the repository must reproduce them.

## License

MIT.
