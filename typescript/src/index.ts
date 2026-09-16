/**
 * jsontensors keeps a JSON document's arrays out of the text and in a binary
 * buffer behind it, with each array in its place in the document and no copy
 * on read.
 *
 * A file is an eight-byte little-endian header length, that many bytes of
 * UTF-8 JSON padded to an eight-byte boundary, and then the tensors' bytes.
 * In the JSON, each array's place holds a reference naming its dtype, shape,
 * and byte range. This package transcribes the specification in the
 * repository's SPEC.md.
 *
 * This entry point is runtime-neutral: it decodes and encodes bytes. The
 * `jsontensors/node` entry point adds reading, ranged reads, and atomic
 * writes of files.
 */

export { DTYPES, DTYPE_WIDTHS, TYPED_ARRAYS, bf16ToFloat32, dtypeOf, float32ToBf16, isDtype } from "./dtype.ts";
export type { Dtype, TypedArray, TypedArrayOf } from "./dtype.ts";
export { decode, resolve } from "./decode.ts";
export { chunks, encode, layout, writeNumber, writeReference } from "./encode.ts";
export type { Layout } from "./encode.ts";
export { JsontensorsError, NeedMore } from "./errors.ts";
export type { Category } from "./errors.ts";
export { MAX_SAFE_INTEGER, checkTiling, parseHeader, parseJson } from "./header.ts";
export type { Header } from "./header.ts";
export { Tensor, TensorRef, bools, elementCount, mapTensors } from "./value.ts";
export type { Document, Value } from "./value.ts";

/** The file suffix. */
export const SUFFIX = ".jsontensors";

/** The media type. */
export const MEDIA_TYPE = "application/x-jsontensors";
