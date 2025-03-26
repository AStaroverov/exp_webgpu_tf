export function flatFloat32Array(arr: Float32Array[]): Float32Array {
    const out = new Float32Array(arr.reduce((acc, v) => acc + v.length, 0));
    let offset = 0;
    for (const v of arr) {
        out.set(v, offset);
        offset += v.length;
    }
    return out;
}