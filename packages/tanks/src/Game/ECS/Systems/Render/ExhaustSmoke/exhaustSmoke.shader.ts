import { ShaderMeta } from '../../../../../../../renderer/src/WGSL/ShaderMeta.ts';
import { VariableKind, VariableMeta } from '../../../../../../../renderer/src/Struct/VariableMeta.ts';
import { wgsl } from '../../../../../../../renderer/src/WGSL/wgsl.ts';

// Maximum number of exhaust smoke instances
export const MAX_EXHAUST_SMOKE_COUNT = 256;

export const shaderMeta = new ShaderMeta(
    {
        projection: new VariableMeta('uProjection', VariableKind.Uniform, `mat4x4<f32>`),
        // Array of smoke transforms (position/scale encoded in matrix)
        transform: new VariableMeta('uTransform', VariableKind.StorageRead, `array<mat4x4<f32>, ${MAX_EXHAUST_SMOKE_COUNT}>`),
        // Smoke data: progress (0-1), seed
        smokeData: new VariableMeta('uSmokeData', VariableKind.StorageRead, `array<vec2<f32>, ${MAX_EXHAUST_SMOKE_COUNT}>`),
    },
    {},
    // language=WGSL
    wgsl/* wgsl */`
        struct VertexOutput {
            @builtin(position) position: vec4f,
            @location(0) @interpolate(flat) instance_index: u32,
            @location(1) local_position: vec2f,
        };

        @vertex
        fn vs_main(
            @builtin(vertex_index) vertex_index: u32,
            @builtin(instance_index) instance_index: u32,
        ) -> VertexOutput {
            let smokeInfo = uSmokeData[instance_index];
            let progress = smokeInfo.x;
            
            // Smoke expands as it rises and fades
            let expandFactor = 1.0 + progress * 4.0;
            let maxRadius = 1.0 * expandFactor;
            
            // Create quad vertices centered at origin
            let local_pos = vec2f(
                select(-maxRadius, maxRadius, vertex_index > 0u && vertex_index < 4u),
                select(-maxRadius, maxRadius, vertex_index > 1u && vertex_index < 5u)
            );
            
            // Transform to world/screen position
            let world_pos = (uProjection * uTransform[instance_index] * vec4f(local_pos, 0.0, 1.0)).xy;
            
            let position = vec4f(world_pos.x, -world_pos.y, uTransform[instance_index][3].z, 1.0);

            return VertexOutput(
                position,
                instance_index,
                local_pos,
            );
        }

        // Hash function for pseudo-random generation
        fn hash21(p: vec2f) -> f32 {
            var p3 = fract(vec3f(p.xyx) * vec3f(0.1031, 0.1030, 0.0973));
            p3 += dot(p3, p3.yzx + 19.19);
            return fract((p3.x + p3.y) * p3.z);
        }

        // Soft circular gradient
        fn softCircle(uv: vec2f, center: vec2f, radius: f32, softness: f32) -> f32 {
            let dist = length(uv - center);
            return 1.0 - smoothstep(radius - softness, radius + softness, dist);
        }

        // Layered smoke pattern for organic look
        fn smokePattern(localPos: vec2f, progress: f32, seed: f32) -> vec4f {
            let dist = length(localPos);
            
            // Smoke expands over time (stronger expansion)
            let expandFactor = 1.0 + progress * 4.0;
            let currentSize = 1.0 * expandFactor;
            
            if (dist > currentSize) {
                return vec4f(0.0);
            }
            
            // Create turbulent billowing effect with multiple noise layers
            var turbulence = 0.0;
            let angle = atan2(localPos.y, localPos.x);
            
            for (var i = 0u; i < 5u; i++) {
                let freq = f32(i + 2u);
                let noiseAngle = angle * freq + seed * 10.0 + progress * f32(i) * 0.5;
                let noiseMag = hash21(vec2f(seed + f32(i) * 0.17, f32(i) * 0.31));
                turbulence += sin(noiseAngle) * noiseMag * (0.25 / freq);
            }
            
            let noisySize = currentSize * (1.0 + turbulence * 0.5);
            let normalizedDist = dist / noisySize;
            
            if (normalizedDist > 1.0) {
                return vec4f(0.0);
            }
            
            // Dark gray smoke color - varies slightly
            let colorVariation = hash21(vec2f(seed * 3.0, progress * 0.1));
            let baseGray = 0.25 + colorVariation * 0.1;
            let smokeColor = vec3f(baseGray, baseGray * 0.95, baseGray * 0.9);
            
            // Soft edge falloff (stronger falloff)
            let edgeFalloff = 1.0 - smoothstep(0.2, 0.9, normalizedDist);
            
            // Fade out over time (stronger, starts earlier)
            let fadeOut = 1.0 - smoothstep(0.1, 0.7, progress);
            
            // Center is slightly denser
            let centerDensity = 1.0 - normalizedDist * 0.4;
            
            let alpha = edgeFalloff * fadeOut * centerDensity * 0.5;
            
            return vec4f(smokeColor, alpha);
        }

        // Secondary wisp pattern for detail
        fn wispPattern(localPos: vec2f, progress: f32, seed: f32) -> vec4f {
            let wispCount = 4u;
            var totalAlpha = 0.0;
            var wispColor = vec3f(0.3, 0.28, 0.26);
            
            for (var i = 0u; i < wispCount; i++) {
                let wispSeed = hash21(vec2f(seed * 20.0 + f32(i), f32(i) * 7.0));
                let wispAngle = f32(i) * 1.57 + wispSeed * 0.8;
                
                // Wisps drift outward faster
                let driftSpeed = 0.8 + wispSeed * 0.5;
                let wispDist = progress * driftSpeed * (1.0 + progress * 1.5);
                
                let wispPos = vec2f(cos(wispAngle), sin(wispAngle)) * wispDist;
                let distToWisp = length(localPos - wispPos);
                
                // Wisp size grows then shrinks faster
                let wispSize = (0.4 + progress * 0.6) * (1.0 - progress * 0.7);
                
                if (distToWisp < wispSize) {
                    // Stronger fade for wisps
                    let fadeOut = 1.0 - smoothstep(0.2, 0.8, progress);
                    let wispAlpha = (1.0 - distToWisp / wispSize) * fadeOut * 0.3;
                    totalAlpha = max(totalAlpha, wispAlpha);
                }
            }
            
            return vec4f(wispColor, totalAlpha);
        }

        @fragment
        fn fs_main(
            @location(0) @interpolate(flat) instance_index: u32,
            @location(1) local_position: vec2f,
        ) -> @location(0) vec4f {
            let smokeInfo = uSmokeData[instance_index];
            let progress = smokeInfo.x;
            let seed = smokeInfo.y;
            
            var finalColor = vec3f(0.0);
            var totalAlpha = 0.0;
            
            // Main smoke body
            let smoke = smokePattern(local_position, progress, seed);
            if (smoke.a > 0.0) {
                finalColor = smoke.rgb;
                totalAlpha = smoke.a;
            }
            
            // Add wisps for detail
            let wisps = wispPattern(local_position, progress, seed);
            if (wisps.a > 0.0) {
                finalColor = mix(finalColor, wisps.rgb, wisps.a * 0.5);
                totalAlpha = max(totalAlpha, wisps.a);
            }
            
            if (totalAlpha < 0.01) {
                discard;
            }
            
            return vec4f(finalColor, totalAlpha);
        }
    `,
);

