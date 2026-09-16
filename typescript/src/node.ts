/**
 * The file operations, for Node: reading a file whole or by header, ranged
 * reads of tensors, and atomic writes.
 */

import {
  closeSync,
  fstatSync,
  fsyncSync,
  mkdtempSync,
  openSync,
  readFileSync,
  readSync,
  renameSync,
  rmSync,
  writeSync,
} from "node:fs";
import { dirname, join } from "node:path";

import { decode } from "./decode.ts";
import { chunks } from "./encode.ts";
import { JsontensorsError, NeedMore } from "./errors.ts";
import type { Header } from "./header.ts";
import { FIRST_PREFIX, parseHeader } from "./header.ts";
import type { Document } from "./value.ts";
import { Tensor, type TensorRef } from "./value.ts";

/** Reads a file whole and decodes it. Each tensor is a view over the file's bytes in memory. */
export function read(path: string): Document<Tensor> {
  const file = readFileSync(path);
  return decode(new Uint8Array(file.buffer, file.byteOffset, file.byteLength));
}

/** Reads a file's header alone, touching no tensor bytes. */
export function readHeader(path: string): Header {
  const fd = openSync(path, "r");
  try {
    return headerOf(fd);
  } finally {
    closeSync(fd);
  }
}

function headerOf(fd: number): Header {
  const total = fstatSync(fd).size;
  let prefix = new Uint8Array(Math.min(total, FIRST_PREFIX));
  readSync(fd, prefix, 0, prefix.length, 0);
  try {
    return parseHeader(prefix, total);
  } catch (error) {
    if (!(error instanceof NeedMore)) {
      throw error;
    }
    const grown = new Uint8Array(error.required);
    grown.set(prefix);
    readSync(fd, grown, prefix.length, grown.length - prefix.length, prefix.length);
    prefix = grown;
    return parseHeader(prefix, total);
  }
}

/**
 * A file opened for ranged reads: the header parsed once, the tensors left
 * on disk, and any byte range of any tensor read on request with one
 * positioned read.
 */
export class Ranged {
  readonly path: string;
  readonly header: Header;
  private readonly fd: number;

  constructor(path: string) {
    this.path = path;
    this.fd = openSync(path, "r");
    try {
      this.header = headerOf(this.fd);
    } catch (error) {
      closeSync(this.fd);
      throw error;
    }
  }

  /** The document with references in the arrays' places. */
  get document(): Document<TensorRef> {
    return this.header.document;
  }

  /** `length` bytes of a reference starting `start` bytes into it. */
  read(reference: TensorRef, start: number, length: number): Uint8Array {
    if (start < 0 || length < 0 || start + length > reference.length) {
      throw new JsontensorsError(
        `the range [${start}, ${start + length}) lies outside a ${reference.length}-byte reference`,
      );
    }
    const out = new Uint8Array(length);
    let done = 0;
    while (done < length) {
      const n = readSync(this.fd, out, done, length - done, this.header.bufferStart + reference.offset + start + done);
      if (n === 0) {
        throw new JsontensorsError(`${this.path} ended inside a tensor`);
      }
      done += n;
    }
    return out;
  }

  /** A reference's whole tensor, read into memory. */
  tensor(reference: TensorRef): Tensor {
    return new Tensor(reference.dtype, reference.shape, this.read(reference, 0, reference.length));
  }

  close(): void {
    closeSync(this.fd);
  }

  [Symbol.dispose](): void {
    this.close();
  }
}

/** Opens a file for ranged reads. */
export function openRanged(path: string): Ranged {
  return new Ranged(path);
}

/**
 * Writes a document to a path atomically: a temporary file in the target's
 * directory, synced, then renamed over the target.
 */
export function write(path: string, document: Document<Tensor>): void {
  const dir = mkdtempSync(join(dirname(path), ".jsontensors-"));
  const temp = join(dir, "file");
  try {
    const fd = openSync(temp, "w");
    try {
      for (const chunk of chunks(document)) {
        let done = 0;
        while (done < chunk.length) {
          done += writeSync(fd, chunk, done, chunk.length - done);
        }
      }
      fsyncSync(fd);
    } finally {
      closeSync(fd);
    }
    renameSync(temp, path);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}
