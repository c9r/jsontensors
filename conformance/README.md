# Conformance

The cases in `cases` are the contract between the implementations. Each case is a `.jsontensors` file and a JSON file of the same stem saying what an implementation must do with it.

## Valid cases

A valid case has `<stem>.jsontensors` and `<stem>.expected.json`:

```json
{
  "canonical": true,
  "document": { "title": "...", "channels": [{ "name": "left", "samples": null }] },
  "tensors": [
    { "path": "/channels/0/samples", "dtype": "F32", "shape": [96000], "length": 384000, "sha256": "..." }
  ]
}
```

`document` is the decoded document as plain JSON with `null` standing at each tensor's position. `tensors` lists every tensor by its JSON Pointer path, with its dtype, shape, byte length, and the SHA-256 of its bytes. An implementation must decode the file to that document and, for each listed path, to a tensor of that dtype, shape, and byte content. Equality of the document is equality of JSON values.

`canonical` marks a file written by the canonical layout with numbers that print identically in every language. An implementation must reproduce a canonical file byte for byte by decoding it and encoding the result. A file that is not canonical, because its header is not compact, its numbers have several spellings, or its layout is not the writer's, must still decode to the stated values and must round-trip by value through the implementation's own encoder.

## Invalid cases

An invalid case has `<stem>.jsontensors` and `<stem>.error.json`:

```json
{ "error": "tiling" }
```

An implementation must refuse the file, and its error must be of the named category. Where a strict and a lenient JSON parser would legitimately classify a file differently, `error` lists every acceptable category. The categories are:

| Category | Meaning |
| --- | --- |
| `truncated-length` | the file ends before the eight length bytes |
| `truncated-header` | the header length overruns the file |
| `invalid-json` | the header is not UTF-8 JSON |
| `non-object-root` | the header parses but its root is not an object |
| `malformed-reference` | an object with `$dtype` is not a well-formed reference |
| `length-mismatch` | a reference's length disagrees with its shape and dtype |
| `stray-dollar` | a single-`$` property name outside a reference |
| `tiling` | the references do not tile the buffer |
| `number` | a number that is not exactly representable as a double |

## Generating the cases

The Rust implementation writes the cases with `cargo run --example generate` from `rust/`. The valid files come from its encoder, and the invalid ones are assembled byte by byte, since no encoder produces them. Regenerating must reproduce every file exactly.
