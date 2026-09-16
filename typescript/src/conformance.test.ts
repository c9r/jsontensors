import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { suite, test } from "node:test";

import { decode, encode } from "./index.ts";
import type { Document, Value } from "./index.ts";
import { Tensor } from "./index.ts";

const CASES = join(import.meta.dirname, "../../conformance/cases");

interface Materialized {
  readonly document: unknown;
  readonly tensors: readonly unknown[];
}

/** The decoded document in the expectation's form: `null` at each tensor, and the tensors listed by pointer. */
function materialize(document: Document<Tensor>): Materialized {
  const tensors: unknown[] = [];
  const walk = (value: Value<Tensor>, path: string): unknown => {
    if (value instanceof Tensor) {
      tensors.push({
        path,
        dtype: value.dtype,
        shape: [...value.shape],
        length: value.bytes.length,
        sha256: createHash("sha256").update(value.bytes).digest("hex"),
      });
      return null;
    }
    if (Array.isArray(value)) {
      return (value as readonly Value<Tensor>[]).map((item, index) => walk(item, `${path}/${index}`));
    }
    if (value !== null && typeof value === "object") {
      const out: Record<string, unknown> = {};
      for (const [key, item] of Object.entries(value as { readonly [key: string]: Value<Tensor> })) {
        Object.defineProperty(out, key, {
          value: walk(item, path + "/" + key.replaceAll("~", "~0").replaceAll("/", "~1")),
          enumerable: true,
          writable: true,
          configurable: true,
        });
      }
      return out;
    }
    return value;
  };
  return { document: walk(document, ""), tensors };
}

/** Equality of JSON values, with property order significant and every number a double. */
function same(a: unknown, b: unknown): boolean {
  if (Array.isArray(a) || Array.isArray(b)) {
    return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((x, i) => same(x, b[i]));
  }
  if (a !== null && typeof a === "object" && b !== null && typeof b === "object") {
    const ka = Object.keys(a);
    const kb = Object.keys(b);
    return (
      ka.length === kb.length &&
      ka.every((k, i) => k === kb[i] && same((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]))
    );
  }
  return Object.is(a, b) || (typeof a === "number" && typeof b === "number" && a === b);
}

void suite("conformance", () => {
  const stems = readdirSync(CASES)
    .filter((name) => name.endsWith(".jsontensors"))
    .map((name) => name.slice(0, -".jsontensors".length))
    .sort();
  assert.ok(stems.length > 0, `no cases in ${CASES}; run \`cargo run --example generate\` in rust/`);

  for (const stem of stems) {
    void test(stem, () => {
      const data = new Uint8Array(readFileSync(join(CASES, `${stem}.jsontensors`)));
      const expectedPath = join(CASES, `${stem}.expected.json`);
      const errorPath = join(CASES, `${stem}.error.json`);
      let expectedText: string | undefined;
      let errorText: string | undefined;
      try {
        expectedText = readFileSync(expectedPath, "utf8");
      } catch {
        errorText = readFileSync(errorPath, "utf8");
      }
      if (expectedText !== undefined) {
        const expected = JSON.parse(expectedText) as { canonical: boolean; document: unknown; tensors: unknown[] };
        const decoded = decode(data);
        const got = materialize(decoded);
        assert.ok(same(got.document, expected.document), `decoded document differs: ${JSON.stringify(got.document)}`);
        assert.ok(same(got.tensors, expected.tensors), `tensors differ: ${JSON.stringify(got.tensors)}`);
        const reEncoded = encode(decoded);
        if (expected.canonical) {
          assert.deepEqual(reEncoded, data, "a canonical case did not reproduce byte for byte");
        } else {
          assert.ok(same(materialize(decode(reEncoded)), got), "the re-encoding decoded to different values");
        }
      } else {
        const spec = JSON.parse(errorText!) as { error: string | string[] };
        const accepted = typeof spec.error === "string" ? [spec.error] : spec.error;
        assert.throws(
          () => decode(data),
          (error: unknown) => {
            assert.ok(error instanceof Error, "threw something that is not an Error");
            const category = (error as { category?: string }).category;
            assert.ok(
              category !== undefined && accepted.includes(category),
              `refused as ${category}, expected ${accepted.join(" or ")}: ${error.message}`,
            );
            return true;
          },
        );
      }
    });
  }
});
