import { DTYPE_WIDTHS } from "./dtype.ts";
import { JsontensorsError } from "./errors.ts";
import { MAX_SAFE_INTEGER, pointer } from "./header.ts";
import type { Document, Value } from "./value.ts";
import { Tensor, type TensorRef } from "./value.ts";

/** A document laid out for writing: the head and the tensors in buffer order. */
export interface Layout {
  /** The length prefix and the padded header JSON. */
  readonly head: Uint8Array;
  /** The tensors in the order their bytes follow the head. */
  readonly tensors: readonly Tensor[];
  /** The buffer's byte count, which is the sum of the tensors' lengths. */
  readonly bufferLength: number;
  /** The whole file's byte count. */
  readonly totalLength: number;
}

const TEXT = new TextEncoder();

/**
 * Lays a document out: tensors collected in document order, placed widest
 * dtype first with ties in collection order, and the header serialized with
 * each tensor's reference in its place.
 */
export function layout(document: Document<Tensor>): Layout {
  if (document === null || typeof document !== "object" || Array.isArray(document) || document instanceof Tensor) {
    throw new JsontensorsError("the document is not a JSON object");
  }
  const tensors: Tensor[] = [];
  collect(document, [], tensors);

  const order = tensors
    .map((_, i) => i)
    .sort((a, b) => DTYPE_WIDTHS[tensors[b]!.dtype] - DTYPE_WIDTHS[tensors[a]!.dtype] || a - b);
  const references: TensorRef[] = new Array<TensorRef>(tensors.length);
  let offset = 0;
  for (const i of order) {
    references[i] = tensors[i]!.reference(offset);
    offset += tensors[i]!.bytes.length;
  }

  let next = 0;
  const json = writeObject(document, () => references[next++]!);
  const blob = TEXT.encode(json);
  const padding = (8 - ((8 + blob.length) % 8)) % 8;
  const head = new Uint8Array(8 + blob.length + padding);
  new DataView(head.buffer).setBigUint64(0, BigInt(blob.length + padding), true);
  head.set(blob, 8);
  head.fill(0x20, 8 + blob.length);

  return { head, tensors: order.map((i) => tensors[i]!), bufferLength: offset, totalLength: head.length + offset };
}

/** Collects tensors in document order and refuses values JSON cannot carry. */
function collect(value: Value<Tensor>, path: (string | number)[], tensors: Tensor[]): void {
  if (value instanceof Tensor) {
    tensors.push(value);
  } else if (Array.isArray(value)) {
    (value as readonly Value<Tensor>[]).forEach((item, index) => {
      path.push(index);
      collect(item, path, tensors);
      path.pop();
    });
  } else if (value !== null && typeof value === "object") {
    const proto: unknown = Object.getPrototypeOf(value);
    if (proto !== Object.prototype && proto !== null) {
      throw new JsontensorsError(
        `the value at ${pointer(path)} is a ${(value as object).constructor.name}, which is neither JSON nor a tensor`,
      );
    }
    for (const [key, item] of Object.entries(value as { readonly [key: string]: Value<Tensor> })) {
      path.push(key);
      collect(item, path, tensors);
      path.pop();
    }
  } else if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new JsontensorsError(`the number at ${pointer(path)} is not finite, and JSON cannot spell it`);
    }
  } else if (value !== null && typeof value !== "boolean" && typeof value !== "string") {
    throw new JsontensorsError(
      `the value at ${pointer(path)} is a ${typeof value}, which is neither JSON nor a tensor`,
    );
  }
}

function writeValue(value: Value<Tensor>, next: () => TensorRef): string {
  if (value instanceof Tensor) {
    return writeReference(next());
  }
  if (value === null) {
    return "null";
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (typeof value === "number") {
    return writeNumber(value);
  }
  if (typeof value === "string") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return "[" + (value as readonly Value<Tensor>[]).map((item) => writeValue(item, next)).join(",") + "]";
  }
  return writeObject(value as { readonly [key: string]: Value<Tensor> }, next);
}

function writeObject(object: { readonly [key: string]: Value<Tensor> }, next: () => TensorRef): string {
  const parts: string[] = [];
  for (const [key, item] of Object.entries(object)) {
    parts.push(JSON.stringify(key.startsWith("$") ? "$" + key : key) + ":" + writeValue(item, next));
  }
  return "{" + parts.join(",") + "}";
}

/** A reference in its canonical form and property order. */
export function writeReference(reference: TensorRef): string {
  return `{"$dtype":"${reference.dtype}","shape":[${reference.shape.join(",")}],"offset":${reference.offset},"length":${reference.length}}`;
}

/**
 * A finite double in its shortest round-trip form. An integer-valued double
 * beyond ±2^53 is written in exponent form, so no reader takes it for an
 * integer literal claiming an exactness no double holds.
 */
export function writeNumber(value: number): string {
  if (Number.isInteger(value) && Math.abs(value) > MAX_SAFE_INTEGER) {
    return value.toExponential();
  }
  return String(value);
}

/** The whole file as bytes. */
export function encode(document: Document<Tensor>): Uint8Array {
  const laid = layout(document);
  const out = new Uint8Array(laid.totalLength);
  out.set(laid.head, 0);
  let position = laid.head.length;
  for (const tensor of laid.tensors) {
    out.set(tensor.bytes, position);
    position += tensor.bytes.length;
  }
  return out;
}

/** The file's chunks in order, head first, for a writer that streams. */
export function* chunks(document: Document<Tensor>): Generator<Uint8Array, void, undefined> {
  const laid = layout(document);
  yield laid.head;
  for (const tensor of laid.tensors) {
    yield tensor.bytes;
  }
}
