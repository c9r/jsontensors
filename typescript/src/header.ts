import { DTYPE_WIDTHS, isDtype } from "./dtype.ts";
import { JsontensorsError, NeedMore } from "./errors.ts";
import type { Document } from "./value.ts";
import { TensorRef, elementCount, setOwn } from "./value.ts";

/** The largest integer magnitude a double represents exactly. */
export const MAX_SAFE_INTEGER = 2 ** 53;

const MAX_SAFE_BIGINT = 2n ** 53n;

/** The prefix a reader fetches first, before it knows the header's length. */
export const FIRST_PREFIX = 4096;

/** A parsed header: the document with references in the arrays' places, and where the buffer is. */
export interface Header {
  readonly document: Document<TensorRef>;
  /** The header's byte count, padding included, as the length prefix states it. */
  readonly headerLength: number;
  /** The buffer's first byte, counted from the start of the file. */
  readonly bufferStart: number;
  /** The buffer's byte count, which the references must tile exactly. */
  readonly bufferLength: number;
}

/** The header length a prefix declares, checked against the file's total size. */
export function declaredLength(prefix: Uint8Array, total: number): number {
  if (total < 8) {
    throw new JsontensorsError("the file ends before the eight length bytes", "truncated-length");
  }
  if (prefix.length < 8) {
    throw new NeedMore(Math.min(total, FIRST_PREFIX));
  }
  const view = new DataView(prefix.buffer, prefix.byteOffset, prefix.byteLength);
  const length = view.getBigUint64(0, true);
  if (length > BigInt(total - 8)) {
    throw new JsontensorsError(`the header length ${length} overruns the ${total}-byte file`, "truncated-header");
  }
  return Number(length);
}

/**
 * Parses the header from a prefix of a file whose total size is known.
 *
 * A reader with an object store or a socket learns the size from the store,
 * reads a first prefix, and retries with the size a `NeedMore` names until the
 * header is in hand. The references validate against the total, so a
 * truncated object fails here exactly as a truncated file does.
 */
export function parseHeader(prefix: Uint8Array, total: number): Header {
  const headerLength = declaredLength(prefix, total);
  const end = 8 + headerLength;
  if (prefix.length < end) {
    throw new NeedMore(end);
  }
  const bufferLength = total - end;
  const document = parseJson(prefix.subarray(8, end), bufferLength);
  return { document, headerLength, bufferStart: end, bufferLength };
}

/** Parses the header JSON and validates its references against a buffer of the given length. */
export function parseJson(blob: Uint8Array, bufferLength: number): Document<TensorRef> {
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(blob);
  } catch {
    throw new JsontensorsError("the header is not UTF-8", "invalid-json");
  }
  let raw: unknown;
  try {
    raw = JSON.parse(text, reviveNumbers);
  } catch (error) {
    if (error instanceof JsontensorsError) {
      throw error;
    }
    throw new JsontensorsError(`the header is not JSON: ${(error as Error).message}`, "invalid-json");
  }
  if (raw === null || typeof raw !== "object" || Array.isArray(raw)) {
    throw new JsontensorsError("the header's root is not an object", "non-object-root");
  }
  if (Object.hasOwn(raw, "$dtype")) {
    throw new JsontensorsError("the header's root is a reference, not an object", "non-object-root");
  }
  const references: TensorRef[] = [];
  const document = convertObject(raw as Record<string, unknown>, [], references);
  checkTiling(references, bufferLength);
  return document;
}

/** The reviver that refuses numbers no double holds exactly, using the literal's source text. */
function reviveNumbers(this: unknown, _key: string, value: unknown, context?: { source?: string }): unknown {
  if (typeof value !== "number") {
    return value;
  }
  const literal = context?.source ?? String(value);
  if (!Number.isFinite(value)) {
    throw new JsontensorsError(`the number ${literal} overflows a double`, "number");
  }
  if (/^-?\d+$/.test(literal)) {
    const big = BigInt(literal);
    if (big > MAX_SAFE_BIGINT || big < -MAX_SAFE_BIGINT) {
      throw new JsontensorsError(`the integer ${literal} is not exactly representable as a double`, "number");
    }
  }
  return value;
}

type Segment = string | number;

/** A path as a JSON Pointer, or `the root` for the empty path. */
export function pointer(path: readonly Segment[]): string {
  if (path.length === 0) {
    return "the root";
  }
  return path
    .map((p) => "/" + (typeof p === "number" ? String(p) : p.replaceAll("~", "~0").replaceAll("/", "~1")))
    .join("");
}

function convert(raw: unknown, path: Segment[], references: TensorRef[]): unknown {
  if (Array.isArray(raw)) {
    return (raw as unknown[]).map((item, index) => {
      path.push(index);
      const out = convert(item, path, references);
      path.pop();
      return out;
    });
  }
  if (raw !== null && typeof raw === "object") {
    const object = raw as Record<string, unknown>;
    if (Object.hasOwn(object, "$dtype")) {
      const reference = parseReference(object, path);
      references.push(reference);
      return reference;
    }
    return convertObject(object, path, references);
  }
  return raw;
}

function convertObject(object: Record<string, unknown>, path: Segment[], references: TensorRef[]): Document<TensorRef> {
  const out: Record<string, unknown> = {};
  for (const [key, item] of Object.entries(object)) {
    let name = key;
    if (key.startsWith("$$")) {
      name = key.slice(1);
    } else if (key.startsWith("$")) {
      throw new JsontensorsError(
        `the property ${JSON.stringify(key)} at ${pointer(path)} has a single leading $ outside a reference`,
        "stray-dollar",
      );
    }
    path.push(key);
    setOwn(out, name, convert(item, path, references));
    path.pop();
  }
  return out as Document<TensorRef>;
}

const REFERENCE_KEYS = ["$dtype", "shape", "offset", "length"];

function parseReference(object: Record<string, unknown>, path: readonly Segment[]): TensorRef {
  const where = pointer(path);
  const malformed = (reason: string): JsontensorsError =>
    new JsontensorsError(`the reference at ${where} is malformed: ${reason}`, "malformed-reference");
  const keys = Object.keys(object);
  if (keys.length !== 4 || !REFERENCE_KEYS.every((key) => Object.hasOwn(object, key))) {
    throw malformed(`its properties are ${JSON.stringify(keys)}`);
  }
  const { $dtype: dtype, shape, offset, length } = object;
  if (typeof dtype !== "string" || !isDtype(dtype)) {
    throw malformed(`${JSON.stringify(dtype)} is not a dtype`);
  }
  if (!Array.isArray(shape) || !shape.every(isSize)) {
    throw malformed("shape is not a list of non-negative integers");
  }
  if (!isSize(offset)) {
    throw malformed("offset is not a non-negative integer");
  }
  if (!isSize(length)) {
    throw malformed("length is not a non-negative integer");
  }
  const expected = elementCount(shape) * DTYPE_WIDTHS[dtype];
  if (length !== expected) {
    throw new JsontensorsError(
      `the reference at ${where} has length ${length}, but its shape and dtype make ${expected} bytes`,
      "length-mismatch",
    );
  }
  return new TensorRef(dtype, [...shape], offset, length);
}

/** Whether a value is a non-negative integer within the safe range. */
function isSize(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 0 && value <= MAX_SAFE_INTEGER;
}

/** Checks that the references, ordered by offset, tile the buffer exactly. */
export function checkTiling(references: readonly TensorRef[], bufferLength: number): void {
  const ranges = references.map((r): [number, number] => [r.offset, r.end]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  let cursor = 0;
  for (const [begin, end] of ranges) {
    if (begin > cursor) {
      throw new JsontensorsError(
        `the references do not tile the buffer: a gap of ${begin - cursor} bytes before offset ${begin}`,
        "tiling",
      );
    }
    if (begin < cursor) {
      throw new JsontensorsError(
        `the references do not tile the buffer: the reference at offset ${begin} overlaps the one before it`,
        "tiling",
      );
    }
    cursor = end;
  }
  if (cursor !== bufferLength) {
    throw new JsontensorsError(
      `the references do not tile the buffer: they end at ${cursor}, but the buffer is ${bufferLength} bytes`,
      "tiling",
    );
  }
}
