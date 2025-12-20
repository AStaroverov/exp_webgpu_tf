// Permutation table (will be seeded)
let perm: number[] = [];
let permMod8: number[] = [];

/**
 * Seed the noise generator for reproducible results
 */
export function seedNoise(seed: number): void {
    const p = new Array(256);
    for (let i = 0; i < 256; i++) {
        p[i] = i;
    }
    
    // Fisher-Yates shuffle with seed
    let s = seed;
    for (let i = 255; i > 0; i--) {
        s = (s * 1103515245 + 12345) & 0x7fffffff;
        const j = s % (i + 1);
        [p[i], p[j]] = [p[j], p[i]];
    }
    
    // Duplicate for wrapping
    perm = new Array(512);
    permMod8 = new Array(512);
    for (let i = 0; i < 512; i++) {
        perm[i] = p[i & 255];
        permMod8[i] = perm[i] % 8;
    }
}

const F2 = 0.5 * (Math.sqrt(3) - 1);
const G2 = (3 - Math.sqrt(3)) / 6;

// Gradient vectors for 2D simplex noise
const grad2 = [
    [1, 1], [-1, 1], [1, -1], [-1, -1],
    [1, 0], [-1, 0], [0, 1], [0, -1],
];

/**
 * 2D Simplex noise
 * Returns value in range [-1, 1]
 */
export function simplex2(x: number, y: number, seed: number): number {
    seedNoise(seed);
    // Skew input space
    const s = (x + y) * F2;
    const i = Math.floor(x + s);
    const j = Math.floor(y + s);
    
    // Unskew back
    const t = (i + j) * G2;
    const X0 = i - t;
    const Y0 = j - t;
    const x0 = x - X0;
    const y0 = y - Y0;
    
    // Determine simplex
    let i1: number, j1: number;
    if (x0 > y0) {
        i1 = 1; j1 = 0;
    } else {
        i1 = 0; j1 = 1;
    }
    
    const x1 = x0 - i1 + G2;
    const y1 = y0 - j1 + G2;
    const x2 = x0 - 1 + 2 * G2;
    const y2 = y0 - 1 + 2 * G2;
    
    // Hash coordinates
    const ii = i & 255;
    const jj = j & 255;
    
    // Calculate contributions
    let n0 = 0, n1 = 0, n2 = 0;
    
    let t0 = 0.5 - x0 * x0 - y0 * y0;
    if (t0 >= 0) {
        const gi0 = permMod8[ii + perm[jj]];
        t0 *= t0;
        n0 = t0 * t0 * (grad2[gi0][0] * x0 + grad2[gi0][1] * y0);
    }
    
    let t1 = 0.5 - x1 * x1 - y1 * y1;
    if (t1 >= 0) {
        const gi1 = permMod8[ii + i1 + perm[jj + j1]];
        t1 *= t1;
        n1 = t1 * t1 * (grad2[gi1][0] * x1 + grad2[gi1][1] * y1);
    }
    
    let t2 = 0.5 - x2 * x2 - y2 * y2;
    if (t2 >= 0) {
        const gi2 = permMod8[ii + 1 + perm[jj + 1]];
        t2 *= t2;
        n2 = t2 * t2 * (grad2[gi2][0] * x2 + grad2[gi2][1] * y2);
    }
    
    // Scale to [-1, 1]
    return 70 * (n0 + n1 + n2);
}

/**
 * Fractal Brownian Motion using simplex noise
 * Returns value in range [0, 1]
 */
export function fbm(x: number, y: number, seed: number, octaves: number = 4, lacunarity: number = 2, gain: number = 0.5): number {
    let value = 0;
    let amplitude = 1;
    let frequency = 1;
    let maxValue = 0;
    
    for (let i = 0; i < octaves; i++) {
        value += amplitude * simplex2(x * frequency, y * frequency, seed);
        maxValue += amplitude;
        amplitude *= gain;
        frequency *= lacunarity;
    }
    
    // Normalize to [0, 1]
    return (value / maxValue + 1) * 0.5;
}
