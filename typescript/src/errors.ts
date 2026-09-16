/** The categories the conformance suite names. A decoding error belongs to exactly one. */
export type Category =
  | "truncated-length"
  | "truncated-header"
  | "invalid-json"
  | "non-object-root"
  | "malformed-reference"
  | "length-mismatch"
  | "stray-dollar"
  | "tiling"
  | "number";

/** Any error the format raises. `category` names the conformance category of a decoding error, or is `undefined`. */
export class JsontensorsError extends Error {
  readonly category: Category | undefined;

  constructor(message: string, category?: Category) {
    super(message);
    this.name = "JsontensorsError";
    this.category = category;
  }
}

/** A prefix ended inside the header of a file that does carry it whole. Retry with at least `required` bytes. */
export class NeedMore extends JsontensorsError {
  readonly required: number;

  constructor(required: number) {
    super(`the header needs ${required} bytes`);
    this.name = "NeedMore";
    this.required = required;
  }
}
