/**
 * Reusable WGSL noise functions for shaders
 * Includes: hash functions, value noise, FBM
 */

// language=WGSL
export const noiseWGSL = /* wgsl */ `
    // Hash function: vec2 -> f32
    fn hash21(p: vec2f) -> f32 {
        var p3 = fract(vec3f(p.xyx) * vec3f(0.1031, 0.1030, 0.0973));
        p3 += dot(p3, p3.yzx + 19.19);
        return fract((p3.x + p3.y) * p3.z);
    }

    // Hash function: vec2 -> vec2
    fn hash22(p: vec2f) -> vec2f {
        var p3 = fract(vec3f(p.xyx) * vec3f(0.1031, 0.1030, 0.0973));
        p3 += dot(p3, p3.yzx + 33.33);
        return fract((p3.xx + p3.yz) * p3.zy);
    }

    // Value Noise with Hermite smoothstep interpolation
    fn valueNoise(p: vec2f) -> f32 {
        let i = floor(p);
        let f = fract(p);
        let a = hash21(i);
        let b = hash21(i + vec2f(1.0, 0.0));
        let c = hash21(i + vec2f(0.0, 1.0));
        let d = hash21(i + vec2f(1.0, 1.0));
        let u = f * f * (3.0 - 2.0 * f); // Hermite smoothstep
        return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
    }

    // Fractal Brownian Motion - 4 octaves
    fn fbm4(p: vec2f) -> f32 {
        var v = 0.0;
        var a = 0.5;
        var p_mut = p;
        for (var i = 0; i < 4; i++) {
            v += a * valueNoise(p_mut);
            p_mut = p_mut * 2.0;
            a *= 0.5;
        }
        return v;
    }

    // Fractal Brownian Motion - configurable octaves (up to 8)
    fn fbm(p: vec2f, octaves: i32) -> f32 {
        var v = 0.0;
        var a = 0.5;
        var p_mut = p;
        for (var i = 0; i < octaves; i++) {
            v += a * valueNoise(p_mut);
            p_mut = p_mut * 2.0;
            a *= 0.5;
        }
        return v;
    }

    // High-frequency grain/film noise
    fn grainNoise(p: vec2f, intensity: f32) -> f32 {
        return (hash21(p) - 0.5) * intensity;
    }

    // 2D rotation helper
    fn rotate2d(p: vec2f, angle: f32) -> vec2f {
        let s = sin(angle);
        let c = cos(angle);
        return vec2f(
            p.x * c - p.y * s,
            p.x * s + p.y * c
        );
    }
`;

