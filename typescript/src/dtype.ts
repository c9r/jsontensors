/** The dtype table: the element types a tensor can hold, and their widths in bytes. */
export const DTYPE_WIDTHS = {
  F64: 8,
  F32: 4,
  F16: 2,
  BF16: 2,
  I64: 8,
  I32: 4,
  I16: 2,
  I8: 1,
  U64: 8,
  U32: 4,
  U16: 2,
  U8: 1,
  BOOL: 1,
} as const;

export type Dtype = keyof typeof DTYPE_WIDTHS;

/** Every dtype, in the order the table lists them. */
export const DTYPES = Object.keys(DTYPE_WIDTHS) as readonly Dtype[];

/** Whether a string names a dtype. */
export function isDtype(name: string): name is Dtype {
  return Object.hasOwn(DTYPE_WIDTHS, name);
}

/**
 * The typed array a dtype decodes to. `BF16` has no JavaScript type, so it
 * decodes as the `Uint16Array` of its bit patterns, and `BOOL` as a
 * `Uint8Array` of zeros and ones.
 */
export type TypedArrayOf<D extends Dtype> = D extends "F64"
  ? Float64Array
  : D extends "F32"
    ? Float32Array
    : D extends "F16"
      ? Float16Array
      : D extends "BF16"
        ? Uint16Array
        : D extends "I64"
          ? BigInt64Array
          : D extends "I32"
            ? Int32Array
            : D extends "I16"
              ? Int16Array
              : D extends "I8"
                ? Int8Array
                : D extends "U64"
                  ? BigUint64Array
                  : D extends "U32"
                    ? Uint32Array
                    : D extends "U16"
                      ? Uint16Array
                      : D extends "U8"
                        ? Uint8Array
                        : D extends "BOOL"
                          ? Uint8Array
                          : never;

export type TypedArray = TypedArrayOf<Dtype>;

interface TypedArrayConstructor {
  readonly BYTES_PER_ELEMENT: number;
  new (buffer: ArrayBufferLike, byteOffset?: number, length?: number): TypedArray;
}

export const TYPED_ARRAYS: Record<Dtype, TypedArrayConstructor> = {
  F64: Float64Array,
  F32: Float32Array,
  F16: Float16Array,
  BF16: Uint16Array,
  I64: BigInt64Array,
  I32: Int32Array,
  I16: Int16Array,
  I8: Int8Array,
  U64: BigUint64Array,
  U32: Uint32Array,
  U16: Uint16Array,
  U8: Uint8Array,
  BOOL: Uint8Array,
};

/** The dtype a typed array naturally holds, with `Uint16Array` taken as `U16` and `Uint8Array` as `U8`. */
export function dtypeOf(array: TypedArray): Dtype {
  if (array instanceof Float64Array) return "F64";
  if (array instanceof Float32Array) return "F32";
  if (array instanceof Float16Array) return "F16";
  if (array instanceof BigInt64Array) return "I64";
  if (array instanceof Int32Array) return "I32";
  if (array instanceof Int16Array) return "I16";
  if (array instanceof Int8Array) return "I8";
  if (array instanceof BigUint64Array) return "U64";
  if (array instanceof Uint32Array) return "U32";
  if (array instanceof Uint16Array) return "U16";
  return "U8";
}

/** The `bfloat16` bit patterns of a `Uint16Array` as single-precision floats. */
export function bf16ToFloat32(bits: Uint16Array): Float32Array {
  const out = new Float32Array(bits.length);
  const view = new DataView(out.buffer);
  for (let i = 0; i < bits.length; i++) {
    view.setUint32(i * 4, bits[i]! << 16, true);
  }
  return out;
}

/** Single-precision floats as `bfloat16` bit patterns, rounded to nearest even. */
export function float32ToBf16(values: Float32Array): Uint16Array {
  const out = new Uint16Array(values.length);
  const bits = new Uint32Array(values.buffer, values.byteOffset, values.length);
  for (let i = 0; i < values.length; i++) {
    const word = bits[i]!;
    if ((word & 0x7fffffff) > 0x7f800000) {
      out[i] = (word >>> 16) | 0x40;
      continue;
    }
    const lower = word & 0xffff;
    const upper = word >>> 16;
    const roundUp = lower > 0x8000 || (lower === 0x8000 && (upper & 1) === 1);
    out[i] = (upper + (roundUp ? 1 : 0)) & 0xffff;
  }
  return out;
}
