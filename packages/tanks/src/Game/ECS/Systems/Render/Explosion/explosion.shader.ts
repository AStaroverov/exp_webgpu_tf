import { ShaderMeta } from '../../../../../../../renderer/src/WGSL/ShaderMeta.ts';
import { VariableKind, VariableMeta } from '../../../../../../../renderer/src/Struct/VariableMeta.ts';
import { wgsl } from '../../../../../../../renderer/src/WGSL/wgsl.ts';

// Maximum number of explosion instances
export const MAX_EXPLOSION_COUNT = 32;

export const shaderMeta = new ShaderMeta(
    {
        projection: new VariableMeta('uProjection', VariableKind.Uniform, `mat4x4<f32>`),
        // Array of explosion transforms (position encoded in matrix)
        transform: new VariableMeta('uTransform', VariableKind.StorageRead, `array<mat4x4<f32>, ${MAX_EXPLOSION_COUNT}>`),
        // Explosion data: size, progress (0-1), seed, unused
        explosionData: new VariableMeta('uExplosionData', VariableKind.StorageRead, `array<vec4<f32>, ${MAX_EXPLOSION_COUNT}>`),
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
            let explosionInfo = uExplosionData[instance_index];
            let size = explosionInfo.x;
            let progress = explosionInfo.y;
            
            // Explosion expands then contracts slightly
            let expandFactor = 1.0 + progress * 0.5;
            let maxRadius = size * expandFactor * 1.5;
            
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

        // Fireball pattern - main explosion body
        fn fireballPattern(localPos: vec2f, size: f32, progress: f32, seed: f32) -> vec4f {
            let dist = length(localPos);
            
            // Explosion expands rapidly then fades
            let expandProgress = min(progress * 3.0, 1.0);
            let currentSize = size * (0.3 + expandProgress * 0.7);
            
            if (dist > currentSize) {
                return vec4f(0.0);
            }
            
            // Turbulent edge using noise
            let angle = atan2(localPos.y, localPos.x);
            var edgeNoise = 0.0;
            for (var i = 0u; i < 6u; i++) {
                let noiseAngle = angle * (f32(i) + 3.0) + seed * 10.0 + progress * 2.0;
                let noiseMag = hash21(vec2f(seed + f32(i) * 0.1, f32(i)));
                edgeNoise += sin(noiseAngle) * noiseMag * 0.15;
            }
            
            let noisySize = currentSize * (1.0 + edgeNoise);
            let normalizedDist = dist / noisySize;
            
            if (normalizedDist > 1.0) {
                return vec4f(0.0);
            }
            
            // Color gradient from center (white/yellow) to edge (orange/red)
            let innerColor = vec3f(1.0, 0.95, 0.7);  // Bright yellow-white
            let midColor = vec3f(1.0, 0.6, 0.1);     // Orange
            let outerColor = vec3f(0.8, 0.2, 0.0);   // Red-orange
            
            var color: vec3f;
            if (normalizedDist < 0.3) {
                color = mix(innerColor, midColor, normalizedDist / 0.3);
            } else {
                color = mix(midColor, outerColor, (normalizedDist - 0.3) / 0.7);
            }
            
            // Intensity falls off from center and over time
            let centerIntensity = 1.0 - normalizedDist * 0.5;
            let fadeOut = 1.0 - smoothstep(0.3, 1.0, progress);
            let intensity = centerIntensity * fadeOut;
            
            return vec4f(color, intensity);
        }

        // Smoke ring that expands outward
        fn smokePattern(localPos: vec2f, size: f32, progress: f32, seed: f32) -> vec4f {
            // Smoke appears after initial flash
            if (progress < 0.1) {
                return vec4f(0.0);
            }
            
            let smokeProgress = (progress - 0.1) / 0.9;
            let dist = length(localPos);
            
            // Smoke ring expands outward
            let ringRadius = size * (0.5 + smokeProgress * 0.8);
            let ringWidth = size * 0.4 * (1.0 - smokeProgress * 0.3);
            
            let ringDist = abs(dist - ringRadius);
            
            if (ringDist > ringWidth) {
                return vec4f(0.0);
            }
            
            // Add noise to smoke
            let angle = atan2(localPos.y, localPos.x);
            var noise = 0.0;
            for (var i = 0u; i < 4u; i++) {
                let noiseAngle = angle * (f32(i) + 2.0) + seed * 5.0;
                noise += sin(noiseAngle + smokeProgress * 3.0) * 0.2;
            }
            
            let smokeIntensity = (1.0 - ringDist / ringWidth) * (1.0 - smokeProgress);
            let finalIntensity = smokeIntensity * (0.8 + noise * 0.2);
            
            // Dark gray smoke
            let smokeColor = vec3f(0.3, 0.28, 0.25) * (1.0 - smokeProgress * 0.5);
            
            return vec4f(smokeColor, max(0.0, finalIntensity * 0.6));
        }

        // Debris/sparks flying outward
        fn debrisPattern(localPos: vec2f, size: f32, progress: f32, seed: f32) -> vec4f {
            let dist = length(localPos);
            let angle = atan2(localPos.y, localPos.x);
            
            var debrisIntensity = 0.0;
            var debrisColor = vec3f(1.0, 0.7, 0.3);
            
            let debrisCount = 16u;
            for (var i = 0u; i < debrisCount; i++) {
                let debrisSeed = hash21(vec2f(seed * 50.0 + f32(i), f32(i) * 13.0));
                let debrisAngle = f32(i) * 6.28318 / f32(debrisCount) + debrisSeed * 0.8;
                
                // Debris speed varies
                let speed = size * (1.5 + debrisSeed * 1.5);
                let debrisDist = progress * speed;
                
                // Debris position
                let debrisPos = vec2f(cos(debrisAngle), sin(debrisAngle)) * debrisDist;
                let distToDebris = length(localPos - debrisPos);
                
                // Debris size shrinks over time
                let debrisSize = size * 0.08 * (1.0 - progress * 0.7);
                
                if (distToDebris < debrisSize) {
                    let brightness = (1.0 - distToDebris / debrisSize) * (1.0 - progress);
                    debrisIntensity = max(debrisIntensity, brightness);
                    
                    // Color varies from yellow to orange
                    let colorMix = hash21(vec2f(debrisSeed, seed));
                    debrisColor = mix(
                        vec3f(1.0, 0.9, 0.4),
                        vec3f(1.0, 0.5, 0.1),
                        colorMix
                    );
                }
            }
            
            return vec4f(debrisColor, debrisIntensity);
        }

        // Initial bright flash
        fn flashPattern(localPos: vec2f, size: f32, progress: f32, seed: f32) -> vec4f {
            // Flash only in first 15% of animation
            if (progress > 0.15) {
                return vec4f(0.0);
            }
            
            let flashProgress = progress / 0.15;
            let dist = length(localPos);
            
            // Flash expands very quickly
            let flashSize = size * (0.5 + flashProgress * 1.0);
            
            if (dist > flashSize) {
                return vec4f(0.0);
            }
            
            let intensity = (1.0 - dist / flashSize) * (1.0 - flashProgress * flashProgress);
            
            // Pure white flash
            return vec4f(1.0, 1.0, 1.0, intensity);
        }

        @fragment
        fn fs_main(
            @location(0) @interpolate(flat) instance_index: u32,
            @location(1) local_position: vec2f,
        ) -> @location(0) vec4f {
            let explosionInfo = uExplosionData[instance_index];
            let size = explosionInfo.x;
            let progress = explosionInfo.y;
            let seed = explosionInfo.z;
            
            var finalColor = vec3f(0.0);
            var totalAlpha = 0.0;
            
            // Layer effects from back to front
            
            // 1. Smoke (background)
            let smoke = smokePattern(local_position, size, progress, seed);
            if (smoke.a > 0.0) {
                finalColor = mix(finalColor, smoke.rgb, smoke.a);
                totalAlpha = max(totalAlpha, smoke.a);
            }
            
            // 2. Fireball (main body)
            let fireball = fireballPattern(local_position, size, progress, seed);
            if (fireball.a > 0.0) {
                finalColor = mix(finalColor, fireball.rgb, fireball.a);
                totalAlpha = max(totalAlpha, fireball.a);
            }
            
            // 3. Debris/sparks
            let debris = debrisPattern(local_position, size, progress, seed);
            if (debris.a > 0.0) {
                finalColor = mix(finalColor, debris.rgb, debris.a);
                totalAlpha = max(totalAlpha, debris.a);
            }
            
            // 4. Initial flash (foreground)
            let flash = flashPattern(local_position, size, progress, seed);
            if (flash.a > 0.0) {
                finalColor = mix(finalColor, flash.rgb, flash.a);
                totalAlpha = max(totalAlpha, flash.a);
            }
            
            if (totalAlpha < 0.01) {
                discard;
            }
            
            return vec4f(finalColor, totalAlpha);
        }
    `,
);
