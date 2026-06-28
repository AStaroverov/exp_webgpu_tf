export class TypedArray {
  public static f64 = (length: number) => new Float64Array(length);
  public static f32 = (length: number) => new Float32Array(length);
  public static u32 = (length: number) => new Uint32Array(length);
  public static i32 = (length: number) => new Int32Array(length);
  public static i8 = (length: number) => new Int8Array(length);
  public static u8 = (length: number) => new Uint8Array(length);
  public static i16 = (length: number) => new Int16Array(length);

  public static fromBuffer = <T extends ArrayLikeConstructor>(
    kind: T,
    buffer: ArrayBufferLike | ArrayLike<number>,
    offset: number,
    length: number,
  ): T["prototype"] => new kind(buffer as ArrayBuffer, offset, length);
}

type ArrayLikeConstructor =
  | Float64ArrayConstructor
  | Float32ArrayConstructor
  | Uint32ArrayConstructor
  | Int32ArrayConstructor;

export type NestedArrayOpts = { sab: ArrayBufferLike | ArrayLike<number>; byteOffset: number };

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
    this.buffer = opts
      ? TypedArray.fromBuffer(kind, opts.sab, opts.byteOffset, length)
      : new kind(length);
    this.bufferLength = this.buffer.length;
  }

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
