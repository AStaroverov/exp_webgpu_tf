export function hashArray(arr: number[], base = 31, mod = 1e9 + 9): number {
    let hash = 0;
    for (let i = 0; i < arr.length; i++) {
        hash = (hash * base + arr[i]) % mod;
    }
    return hash;
}
