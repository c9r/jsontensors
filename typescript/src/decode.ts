import type { Header } from "./header.ts";
import { parseHeader } from "./header.ts";
import type { Document } from "./value.ts";
import { Tensor, TensorRef, mapTensors } from "./value.ts";

/** Replaces each reference in a parsed header with a tensor over the file's bytes. */
export function resolve(header: Header, file: Uint8Array): Document<Tensor> {
  const isRef = (v: unknown): v is TensorRef => v instanceof TensorRef;
  return mapTensors(header.document, isRef, (reference) => {
    const begin = header.bufferStart + reference.offset;
    return new Tensor(reference.dtype, reference.shape, file.subarray(begin, begin + reference.length));
  }) as Document<Tensor>;
}

/** Decodes a file held in memory. Each tensor's bytes are a view of the given bytes, not a copy. */
export function decode(data: Uint8Array): Document<Tensor> {
  return resolve(parseHeader(data, data.length), data);
}
