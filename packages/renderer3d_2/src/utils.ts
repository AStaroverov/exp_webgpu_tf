export function uniq<T>(arr: T[]): T[] {
  return Array.from(new Set(arr));
}

export function uniqBy<T>(arr: T[], predicate: (v: T) => unknown): T[] {
  return Array.from(
    arr.reduce((map, item) => {
      const key = item === null || item === undefined ? item : predicate(item);

      map.has(key) || map.set(key, item);

      return map;
    }, new Map()),
    ([, v]) => v,
  );
}

export function isNil<T>(v: T | null | undefined): v is null | undefined {
  return v === null || v === undefined;
}

export function notNil<T>(v: T | null | undefined): v is T {
  return v !== null && v !== undefined;
}

export class TypedArray {
  public static f64 = (length: number) => new Float64Array(length);
  public static f32 = (length: number) => new Float32Array(length);
  public static u32 = (length: number) => new Uint32Array(length);
  public static i32 = (length: number) => new Int32Array(length);
  public static i8 = (length: number) => new Int8Array(length);
  public static u8 = (length: number) => new Uint8Array(length);
  public static i16 = (length: number) => new Int16Array(length);

  // SAB-backed variant: build a typed-array VIEW over an existing (Shared)ArrayBuffer
  // at a byte offset instead of allocating private memory. Used for bridge columns
  // that must be visible in both the render (main) and physics (worker) threads —
  // the SAB is allocated once by the registry and both threads bind views into it
  // at byte-identical offsets (see sab/registry.ts). Plain ArrayBuffer also works,
  // which keeps node tests and non-isolated paths usable.
  public static fromSAB = <T extends ArrayLikeConstructor>(
    kind: T,
    buffer: ArrayBufferLike,
    byteOffset: number,
    length: number,
    // TS's TypedArray ctor types accept only ArrayBuffer; SharedArrayBuffer is a
    // valid runtime backing store (that is the whole point), so widen the param.
  ): T["prototype"] => new kind(buffer as ArrayBuffer, byteOffset, length);
}

type ArrayLikeConstructor =
  | Float64ArrayConstructor
  | Float32ArrayConstructor
  | Uint32ArrayConstructor
  | Int32ArrayConstructor;

// When `sab` is present the NestedArray is a VIEW into shared memory at `byteOffset`
// (cross-thread bridge column); otherwise it owns a freshly-allocated private buffer
// and may be seeded. The two are mutually exclusive — pass one or the other.
export type NestedArrayOpts =
  | { sab: ArrayBufferLike; byteOffset: number }
  | ArrayLike<number>; // seed (back-compat positional form)

function isSabOpts(o: unknown): o is { sab: ArrayBufferLike; byteOffset: number } {
  return typeof o === "object" && o !== null && "sab" in o;
}

export class NestedArray<T extends ArrayLikeConstructor> {
  public buffer: T["prototype"];
  public bufferLength: number;

  constructor(
    kind: T,
    public batchLength: number,
    public batchCount: number,
    opts?: NestedArrayOpts,
  ) {
    const length = batchLength * batchCount;
    if (isSabOpts(opts)) {
      // SAB path: point this.buffer at a view over shared memory. Every accessor
      // below only indexes this.buffer, so nothing else changes — getBatch's
      // subarray over a SAB-backed view is itself SAB-backed (zero-copy).
      this.buffer = TypedArray.fromSAB(kind, opts.sab, opts.byteOffset, length);
    } else {
      this.buffer = new kind(length);
      if (opts) this.buffer.set(opts);
    }
    this.bufferLength = this.buffer.length;
  }

  // `opts` is either a seed (private buffer) or { sab, byteOffset } (shared view).
  public static f64 = (batchLength: number, batchCount: number, opts?: NestedArrayOpts) =>
    new NestedArray(Float64Array, batchLength, batchCount, opts);

  public static f32 = (batchLength: number, batchCount: number, opts?: NestedArrayOpts) =>
    new NestedArray(Float32Array, batchLength, batchCount, opts);

  public static f16 = (batchLength: number, batchCount: number, opts?: NestedArrayOpts) =>
    new NestedArray(Float32Array, batchLength, batchCount, opts);

  public static u32 = (batchLength: number, batchCount: number, opts?: NestedArrayOpts) =>
    new NestedArray(Uint32Array, batchLength, batchCount, opts);

  public static i32 = (batchLength: number, batchCount: number, opts?: NestedArrayOpts) =>
    new NestedArray(Int32Array, batchLength, batchCount, opts);

  destroy() {
    // @ts-ignore
    this.buffer = null;
    this.bufferLength = 0;
  }

  public getStartOffset(batchIndex: number) {
    return batchIndex * this.batchLength;
  }

  public getEndOffset(batchIndex: number) {
    return (batchIndex + 1) * this.batchLength;
  }

  public get(batchIndex: number, index: number): number {
    return this.buffer[this.getStartOffset(batchIndex) + index];
  }

  public set(batchIndex: number, index: number, value: number) {
    this.buffer[this.getStartOffset(batchIndex) + index] = value;
    return this;
  }

  public fill(value: number) {
    this.buffer.fill(value);
    return this;
  }

  public setBatch(batchIndex: number, values: ArrayLike<number>) {
    this.buffer.set(values, this.getStartOffset(batchIndex));
    return this;
  }

  public getBatch(batchStart: number): T["prototype"] {
    return this.buffer.subarray(this.getStartOffset(batchStart), this.getEndOffset(batchStart));
  }
}
