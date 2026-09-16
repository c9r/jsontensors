# jsontensors

A jsontensors file is a JSON document followed by its tensors. Each tensor's bytes live in a binary buffer after the JSON. The tensor's place in the document holds a reference giving its dtype, shape, and location in the buffer. That is the whole format. It earns its keep on size and on speed. A float costs some twenty bytes as JSON text and four as binary, and reading it as text means parsing it, so a document whose bulk is numeric becomes compact and fast when its arrays leave the text. There is no envelope, no metadata namespace, and no tensor name table. The document owns the top level of the file.

Reading a file is parsing JSON and substituting arrays. Writing one is the inverse. Two rules, substitution and quoting, make the round trip exact.

A file carries the suffix `.jsontensors` and the media type `application/x-jsontensors`.

## Layout

A file is three parts: the length, the header, and the buffer.

- The length is an unsigned 64-bit little-endian integer in the first eight bytes, counting the header's bytes.
- The header is that many bytes of UTF-8 JSON, padded at the end with ASCII spaces so the buffer begins on an 8-byte boundary. The padding counts toward the length, and is legal JSON trailing whitespace.
- The buffer is the rest of the file: the tensors' bytes, contiguous, and nothing else. A document with no tensors has an empty buffer.

Here is the header of a document with two tensors. The references stand where the arrays belong, and everything else sits wherever the document's JSON puts it.

```json
{
  "title": "Rain on a tin roof",
  "sample_rate": 48000,
  "channels": [
    {
      "name": "left",
      "samples": { "$dtype": "F32", "shape": [96000], "offset": 0, "length": 384000 }
    },
    {
      "name": "right",
      "samples": { "$dtype": "F32", "shape": [96000], "offset": 384000, "length": 384000 }
    }
  ],
  "peaks": { "$dtype": "I16", "shape": [2, 1000], "offset": 768000, "length": 4000 }
}
```

## Decoding

A decoder parses the header as JSON and walks the result with two rules.

**Substitution.** Any object with a `$dtype` property is a tensor reference. The decoder replaces it with its array: the referenced buffer bytes, viewed as the stated dtype and shape. A binding hands out a view over the file's bytes wherever its language allows one, so loading a document touches no tensor bytes until a consumer reads them. Consumers copy before mutating.

**Unquoting.** A property name beginning with `$$` loses one `$`.

The walk yields the document: plain JSON values, with arrays standing where the writer put them.

## References

A reference is exactly this object, with these four properties and no others:

```json
{ "$dtype": "F32", "shape": [96000], "offset": 0, "length": 384000 }
```

- `$dtype` names an entry in the dtype table.
- `shape` is the array's shape, a list of non-negative integers. Data is row-major. An empty list is a scalar, an array of one element. A shape with a zero in it is an array of no elements.
- `offset` is the position of the array's first byte, counted from the start of the buffer. Buffer-relative offsets keep references independent of the header's own serialized size.
- `length` is the array's byte count, which must equal the product of the shape's elements times the dtype's width. It is redundant, and stating it anyway makes each reference a self-contained byte range and gives readers a consistency check.

| dtype | width | storage |
| --- | --- | --- |
| `F64` | 8 | IEEE 754 binary64 |
| `F32` | 4 | IEEE 754 binary32 |
| `F16` | 2 | IEEE 754 binary16 |
| `BF16` | 2 | bfloat16, the high half of a binary32 |
| `I64`, `I32`, `I16`, `I8` | 8, 4, 2, 1 | two's-complement signed |
| `U64`, `U32`, `U16`, `U8` | 8, 4, 2, 1 | unsigned |
| `BOOL` | 1 | one byte per element, 0 or 1 |

All multi-byte values are little-endian. A binding exposes each dtype as its language's natural array type, and where the language has no type for one, as the bytes with the dtype named beside them.

## Quoting

Quoting is what makes `$dtype` unforgeable and the format a strict superset of JSON. An encoder adds one `$` to any property name that begins with `$`, and a decoder removes one from any name that begins with `$$`. So `$foo` travels as `$$foo`, `$$foo` as `$$$foo`, and `foo$bar`, whose `$` is not leading, travels as itself. The rule touches property names only, never string values.

A literal `$dtype` property therefore cannot survive encoding, and every `$dtype` in a stored file was written by the format itself. Any JSON document whatsoever round-trips losslessly, making transcoding a bijection over all of JSON. A single leading `$` appears only on `$dtype`, and a reader that finds one anywhere outside a well-formed reference refuses the file.

## Encoding

An encoder inverts decoding. Its input is an ordinary JSON tree whose values may also be tensors, as property values or as array elements. The root must be an object.

- Property names are quoted.
- Tensors are collected in document order: a depth-first walk, properties in document order, array elements in index order.
- The buffer lays the tensors out widest dtype first, ties broken by collection order, each immediately after the previous. Widths descend and the header pads to an 8-byte boundary, so every tensor begins at a multiple of its width and decoded views are aligned.
- Each tensor's place in the document receives its reference.
- The header is serialized canonically, padded, and written after the length. The buffer follows.

An encoder stores tensors as row-major little-endian bytes, normalizing layout and byte order, and refuses an array whose dtype is not in the table.

## Numbers

Every number in the document's JSON must be exactly representable as an IEEE 754 double. An encoder refuses an integer beyond ±2^53, and refuses NaN and the infinities, which JSON cannot spell at all. An encoder whose language has no integer type writes an integer-valued double beyond ±2^53 in exponent form, so no reader mistakes it for an integer literal. The restriction is on JSON text alone. A tensor holds any value of its dtype, NaN and infinities included.

Representable numbers are what make re-serialization safe. A conforming file parses to identical values in every language, so parsing and canonically re-printing a document is value-exact, and a writer can transform one part and re-emit the rest without risk of silently altering it. A decoder refuses the violations its language lets it detect: an integer literal beyond ±2^53, and a float literal that overflows a double.

## Validation

Readers fail loudly, never silently. Each of these is an error:

- a file truncated before the eight length bytes, or a header length that overruns the file. A streaming reader that does not yet know the file's size bounds the length with an implementation cap.
- a header that is not valid UTF-8 JSON, or whose root is not an object.
- a malformed reference: a `$dtype` outside the table, extra or missing properties, a shape that is not a list of non-negative integers, an offset or length that is not a non-negative integer, or a length that disagrees with the shape and dtype.
- a single-`$` property name anywhere outside a reference.
- references that fail to tile the buffer. Ordered by offset, each must begin where the previous ended, the first at zero and the last at the buffer's end, zero-length references included at their cursor position. A gap, an overlap, or a reference past the end of the buffer is refused, so truncation and corruption cannot yield silently wrong arrays.
- a number the decoder can tell is not exactly representable as a double.

## Determinism

Serialization is canonical per language: compact, property order preserved from the in-memory document, numbers printed in the language's shortest round-trip form. Loading a file and saving it unchanged therefore reproduces it byte for byte. Different languages may print different bytes for the same document, since float lexemes and escape choices differ. There is no byte contract across languages. Equality between implementations is equality of decoded values: the same JSON values, and for each tensor the same dtype, shape, and bytes.

## Atomicity

A writer assembles the complete file under a temporary name in the target directory and renames it over the target. The rename never crosses a filesystem, and a crash at any point leaves the previous file intact.

## Conformance

The `conformance` directory of the reference repository holds files every implementation must decode to the stated values or refuse for the stated reason, and files every implementation must reproduce byte for byte after decoding and re-encoding them. An implementation that passes the suite is a jsontensors implementation.
