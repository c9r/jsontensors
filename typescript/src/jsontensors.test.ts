import assert from "node:assert/strict";
import { mkdtempSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { suite, test } from "node:test";

import type { Document } from "./index.ts";
import {
  JsontensorsError,
  NeedMore,
  Tensor,
  TensorRef,
  bf16ToFloat32,
  bools,
  decode,
  encode,
  float32ToBf16,
  layout,
  parseHeader,
  writeNumber,
} from "./index.ts";
import { openRanged, read, readHeader, write } from "./node.ts";

function document(): Document<Tensor> {
  return {
    title: "Rain on a tin roof",
    sample_rate: 48000,
    channels: [
      { name: "left", samples: Tensor.from(Float32Array.from([0.25, -0.5, 1, 2, -2, 0.125, 8, -8])) },
      { name: "right", samples: Tensor.from(new Float32Array(8)) },
    ],
    peaks: Tensor.from(Int16Array.from([1, -1, 3, -3, 0, 7, -7, 100]), [2, 4]),
    flags: bools([true, false, true]),
    scalar: Tensor.from(Float64Array.from([42]), []),
    empty: Tensor.from(new Int32Array(0), [0, 3]),
  };
}

function frame(text: string, buffer: Uint8Array = new Uint8Array(0)): Uint8Array {
  const blob = new TextEncoder().encode(text);
  const padding = (8 - ((8 + blob.length) % 8)) % 8;
  const out = new Uint8Array(8 + blob.length + padding + buffer.length);
  new DataView(out.buffer).setBigUint64(0, BigInt(blob.length + padding), true);
  out.set(blob, 8);
  out.fill(0x20, 8 + blob.length, 8 + blob.length + padding);
  out.set(buffer, 8 + blob.length + padding);
  return out;
}

function headerJson(
  data: Uint8Array,
): Record<string, { $dtype: string; shape: number[]; offset: number; length: number }> {
  const length = Number(new DataView(data.buffer, data.byteOffset).getBigUint64(0, true));
  return JSON.parse(new TextDecoder().decode(data.subarray(8, 8 + length))) as ReturnType<typeof headerJson>;
}

void suite("encode and decode", () => {
  void test("round-trips a document with tensors as views", () => {
    const data = encode(document());
    const decoded = decode(data);
    assert.deepEqual(decoded, document());
    const peaks = decoded["peaks"] as Tensor;
    assert.equal(peaks.bytes.buffer, data.buffer, "a decoded tensor views the file's bytes");
    assert.deepEqual(Array.from(peaks.array() as Int16Array), [1, -1, 3, -3, 0, 7, -7, 100]);
    assert.deepEqual((decoded["flags"] as Tensor).bools(), [true, false, true]);
  });

  void test("lays tensors out widest first with ties in document order", () => {
    const laid = layout({
      b: Tensor.from(Uint8Array.from([1, 2, 3])),
      d: Tensor.from(Float64Array.from([1])),
      s: Tensor.from(Uint16Array.from([1, 2])),
      t: Tensor.from(Int16Array.from([1])),
    });
    assert.deepEqual(
      laid.tensors.map((t) => t.dtype),
      ["F64", "U16", "I16", "U8"],
    );
    assert.equal(laid.head.length % 8, 0);
    const header = headerJson(laid.head);
    assert.equal(header["d"]!.offset, 0);
    assert.equal(header["s"]!.offset, 8);
    assert.equal(header["t"]!.offset, 12);
    assert.equal(header["b"]!.offset, 14);
  });

  void test("quotes leading dollars and round-trips any property name", () => {
    const doc = { $dtype: "data", $$x: null, $: 1, a$b: true, "": "empty" };
    const laid = layout(doc);
    assert.equal(
      new TextDecoder().decode(laid.head.subarray(8)).trimEnd(),
      '{"$$dtype":"data","$$$x":null,"$$":1,"a$b":true,"":"empty"}',
    );
    assert.deepEqual(decode(encode(doc)), doc);
  });

  void test("prints numbers canonically", () => {
    assert.equal(writeNumber(0), "0");
    assert.equal(writeNumber(-3), "-3");
    assert.equal(writeNumber(2 ** 53), "9007199254740992");
    assert.equal(writeNumber(2 ** 53 + 2), "9.007199254740994e+15");
    assert.equal(writeNumber(0.5), "0.5");
    assert.equal(writeNumber(1e-7), "1e-7");
  });

  void test("refuses what JSON cannot carry", () => {
    assert.throws(() => encode({ n: Number.NaN }), /not finite/);
    assert.throws(() => encode({ n: new Date() as unknown as number }), /neither JSON/);
    assert.throws(() => encode({ n: undefined as unknown as number }), /neither JSON/);
  });

  void test("keeps __proto__ as data", () => {
    const decoded = decode(frame('{"__proto__":{"x":1}}'));
    assert.deepEqual(Object.keys(decoded), ["__proto__"]);
    assert.equal(Object.getPrototypeOf(decoded), Object.prototype);
  });

  void test("converts bfloat16 bit patterns", () => {
    const bits = float32ToBf16(Float32Array.from([1, 3.140625, -0.5, Number.NaN]));
    assert.deepEqual(Array.from(bits.subarray(0, 3)), [0x3f80, 0x4049, 0xbf00]);
    assert.deepEqual(Array.from(bf16ToFloat32(bits).subarray(0, 3)), [1, 3.140625, -0.5]);
    assert.ok(Number.isNaN(bf16ToFloat32(bits)[3]));
  });
});

void suite("header parsing", () => {
  void test("asks for more bytes until the header is whole", () => {
    const data = encode({ x: 1 });
    assert.throws(
      () => parseHeader(data.subarray(0, 3), data.length),
      (e: unknown) => e instanceof NeedMore && e.required === data.length,
    );
    assert.throws(
      () => parseHeader(data.subarray(0, 10), data.length),
      (e: unknown) => e instanceof NeedMore,
    );
    assert.deepEqual(parseHeader(data, data.length).document, { x: 1 });
    assert.throws(() => parseHeader(data, data.length - 1), /overruns/);
  });

  for (const [text, buffer, category] of [
    ['{"a":', new Uint8Array(0), "invalid-json"],
    ["[1]", new Uint8Array(0), "non-object-root"],
    ['{"$dtype":"U8","shape":[1],"offset":0,"length":1}', new Uint8Array(1), "non-object-root"],
    ['{"$x":1}', new Uint8Array(0), "stray-dollar"],
    ['{"a":{"$dtype":"F128","shape":[1],"offset":0,"length":16}}', new Uint8Array(16), "malformed-reference"],
    ['{"a":{"$dtype":"U8","shape":[1],"offset":0}}', new Uint8Array(1), "malformed-reference"],
    ['{"a":{"$dtype":"U8","shape":[2],"offset":0,"length":1}}', new Uint8Array(1), "length-mismatch"],
    ['{"a":{"$dtype":"U8","shape":[2],"offset":1,"length":2}}', new Uint8Array(3), "tiling"],
    ['{"a":{"$dtype":"U8","shape":[2],"offset":0,"length":2}}', new Uint8Array(3), "tiling"],
    ['{"n":9007199254740993}', new Uint8Array(0), "number"],
    ['{"n":1e400}', new Uint8Array(0), "number"],
  ] as const) {
    void test(`refuses ${text} as ${category}`, () => {
      assert.throws(
        () => decode(frame(text, buffer)),
        (e: unknown) => e instanceof JsontensorsError && e.category === category,
      );
    });
  }

  void test("accepts the edges", () => {
    assert.deepEqual(decode(frame('{"n":9007199254740992,"m":-9007199254740992}')), { n: 2 ** 53, m: -(2 ** 53) });
    const scalar = decode(frame('{"s":{"$dtype":"F64","shape":[],"offset":0,"length":8}}', new Uint8Array(8)))[
      "s"
    ] as Tensor;
    assert.deepEqual(scalar.shape, []);
    assert.equal(scalar.elements, 1);
  });
});

void suite("files", () => {
  const dir = mkdtempSync(join(tmpdir(), "jsontensors-"));

  void test("write, read, and read the header", () => {
    const path = join(dir, "doc.jsontensors");
    write(path, document());
    assert.deepEqual(read(path), document());
    const header = readHeader(path);
    assert.deepEqual(header.document["peaks"], new TensorRef("I16", [2, 4], 72, 16));
    assert.equal(header.bufferLength, 8 + 64 + 16 + 3);
    assert.deepEqual(readdirSync(dir), ["doc.jsontensors"], "no temporary file remains");
  });

  void test("ranged reads match the tensor", () => {
    const path = join(dir, "r.jsontensors");
    const values = Uint32Array.from({ length: 2000 }, (_, i) => i);
    write(path, { v: Tensor.from(values), text: "x".repeat(10_000) });
    using ranged = openRanged(path);
    const reference = ranged.document["v"] as TensorRef;
    assert.deepEqual(Array.from(ranged.tensor(reference).array() as Uint32Array), Array.from(values));
    const slice = ranged.read(reference, 400, 8);
    assert.deepEqual(Array.from(new Uint32Array(slice.buffer, slice.byteOffset, 2)), [100, 101]);
    assert.throws(() => ranged.read(reference, 7996, 8), /outside/);
  });

  void test("cleans up", () => {
    rmSync(dir, { recursive: true, force: true });
  });
});
