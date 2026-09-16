import type { Dtype, TypedArray, TypedArrayOf } from "./dtype.ts";
import { DTYPE_WIDTHS, TYPED_ARRAYS, dtypeOf } from "./dtype.ts";
import { JsontensorsError } from "./errors.ts";

/** A JSON value, or a tensor standing where an array belongs. */
export type Value<T> =
  | null
  | boolean
  | number
  | string
  | readonly Value<T>[]
  | { readonly [key: string]: Value<T> }
  | T;

/** A document: the object at the root of a file. */
export type Document<T> = { readonly [key: string]: Value<T> };

/** The element count of a shape, which is one for the empty shape. */
export function elementCount(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

/** A reference: where a tensor's bytes are, without the bytes. */
export class TensorRef {
  readonly dtype: Dtype;
  readonly shape: readonly number[];
  /** The first byte, counted from the start of the buffer. */
  readonly offset: number;
  /** The byte count, which is the element count times the dtype's width. */
  readonly length: number;

  constructor(dtype: Dtype, shape: readonly number[], offset: number, length: number) {
    this.dtype = dtype;
    this.shape = shape;
    this.offset = offset;
    this.length = length;
  }

  /** The element count, which is one for a scalar. */
  get elements(): number {
    return elementCount(this.shape);
  }

  /** The byte after the last, counted from the start of the buffer. */
  get end(): number {
    return this.offset + this.length;
  }
}

/** A tensor with its bytes, which are a view when the source allows one. */
export class Tensor {
  readonly dtype: Dtype;
  readonly shape: readonly number[];
  /** Row-major little-endian bytes, exactly the element count times the width. */
  readonly bytes: Uint8Array;

  constructor(dtype: Dtype, shape: readonly number[], bytes: Uint8Array) {
    const expected = elementCount(shape) * DTYPE_WIDTHS[dtype];
    if (bytes.length !== expected) {
      throw new JsontensorsError(
        `a tensor of dtype ${dtype} and shape [${shape.join(", ")}] takes ${expected} bytes, not ${bytes.length}`,
      );
    }
    this.dtype = dtype;
    this.shape = shape;
    this.bytes = bytes;
  }

  /**
   * A tensor over a typed array's elements, in row-major order, with the
   * shape given or the array's length. `Uint16Array` is `U16` unless `dtype`
   * says `BF16`, and `Uint8Array` is `U8` unless it says `BOOL`.
   */
  static from(array: TypedArray, shape?: readonly number[], dtype?: Dtype): Tensor {
    const kind = dtype ?? dtypeOf(array);
    if (DTYPE_WIDTHS[kind] !== array.BYTES_PER_ELEMENT) {
      throw new JsontensorsError(`${array.constructor.name} does not hold ${kind} elements`);
    }
    const bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
    return new Tensor(kind, shape ?? [array.length], bytes);
  }

  /** The element count. */
  get elements(): number {
    return elementCount(this.shape);
  }

  /** The reference this tensor would have at an offset. */
  reference(offset: number): TensorRef {
    return new TensorRef(this.dtype, this.shape, offset, this.bytes.length);
  }

  /**
   * The elements as the dtype's typed array. A view over the bytes when
   * they are aligned for the dtype, which a file's layout guarantees, and a
   * copy when they are not.
   */
  array(): TypedArrayOf<Dtype> {
    const ctor = TYPED_ARRAYS[this.dtype];
    const length = this.bytes.length / ctor.BYTES_PER_ELEMENT;
    if (this.bytes.byteOffset % ctor.BYTES_PER_ELEMENT === 0) {
      return new ctor(this.bytes.buffer, this.bytes.byteOffset, length);
    }
    const copy = new Uint8Array(this.bytes.length);
    copy.set(this.bytes);
    return new ctor(copy.buffer, 0, length);
  }

  /** The elements of a `BOOL` tensor, refusing any byte that is not 0 or 1. */
  bools(): boolean[] {
    if (this.dtype !== "BOOL") {
      throw new JsontensorsError(`the tensor is ${this.dtype}, not BOOL`);
    }
    return Array.from(this.bytes, (byte) => {
      if (byte > 1) {
        throw new JsontensorsError(`a BOOL tensor holds the byte ${byte}, which is neither 0 nor 1`);
      }
      return byte === 1;
    });
  }
}

/** A `BOOL` tensor from booleans. */
export function bools(values: readonly boolean[], shape?: readonly number[]): Tensor {
  return new Tensor(
    "BOOL",
    shape ?? [values.length],
    Uint8Array.from(values, (b) => (b ? 1 : 0)),
  );
}

/**
 * Define an own property the way `JSON.parse` does, so a key named
 * `__proto__` becomes data instead of a prototype.
 */
export function setOwn<T>(target: Record<string, T>, key: string, value: T): void {
  Object.defineProperty(target, key, { value, enumerable: true, writable: true, configurable: true });
}

/** Whether a value is a JSON array, narrowed without losing the element type. */
export function isArray<T>(value: Value<T>): value is readonly Value<T>[] {
  return Array.isArray(value);
}

/** Rebuilds a value with every tensor replaced through `f`, in document order. */
export function mapTensors<T, U>(value: Value<T>, isTensor: (v: unknown) => v is T, f: (t: T) => U): Value<U> {
  if (isTensor(value)) {
    return f(value);
  }
  if (isArray(value)) {
    return value.map((item) => mapTensors(item, isTensor, f));
  }
  if (value !== null && typeof value === "object") {
    const out: Record<string, Value<U>> = {};
    for (const [key, item] of Object.entries(value)) {
      setOwn(out, key, mapTensors(item, isTensor, f));
    }
    return out;
  }
  return value;
}
