type TypedArrayLike =
    Float64Array |
    Float32Array |
    Uint32Array |
    Int32Array

export function flatTypedArray<T extends TypedArrayLike>(arr: T[]): T {
    const Klass = arr[0].constructor;
    // @ts-expect-error
    const out = new Klass(arr.reduce((acc, v) => acc + v.length, 0));
    let offset = 0;
    for (const v of arr) {
        out.set(v, offset);
        offset += v.length;
    }
    return out;
}