import { ShaderMeta } from '../../../../../../../renderer/src/WGSL/ShaderMeta.ts';
import { VariableKind, VariableMeta } from '../../../../../../../renderer/src/Struct/VariableMeta.ts';
import { wgsl } from '../../../../../../../renderer/src/WGSL/wgsl.ts';

// Maximum number of muzzle flash instances
export const MAX_MUZZLE_FLASH_COUNT = 64;

export const shaderMeta = new ShaderMeta(
    {
        projection: new VariableMeta('uProjection', VariableKind.Uniform, `mat4x4<f32>`),
        // Array of flash transforms (position, rotation, scale encoded in matrix)
        transform: new VariableMeta('uTransform', VariableKind.StorageRead, `array<mat4x4<f32>, ${MAX_MUZZLE_FLASH_COUNT}>`),
        // Flash data: progress (0-1), seed
        flashData: new VariableMeta('uFlashData', VariableKind.StorageRead, `array<vec2<f32>, ${MAX_MUZZLE_FLASH_COUNT}>`),
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
            let flashInfo = uFlashData[instance_index];
            let size = 1.0; // Size is in transform scale
            let progress = flashInfo.x;
            
            // Calculate expanded size for smoke
            let expandedSize = size * (0.6 + progress * 1.44);
            let maxRadius = expandedSize * 2.0;
            
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

        // Hash functions for pseudo-random generation
        fn hash21(p: vec2f) -> f32 {
            var p3 = fract(vec3f(p.xyx) * vec3f(0.1031, 0.1030, 0.0973));
            p3 += dot(p3, p3.yzx + 19.19);
            return fract((p3.x + p3.y) * p3.z);
        }
        
        fn hash22(p: vec2f) -> vec2f {
            var p3 = fract(vec3f(p.xyx) * vec3f(0.1031, 0.1030, 0.0973));
            p3 += dot(p3, p3.yzx + 19.19);
            return fract(vec2f((p3.x + p3.y) * p3.z, (p3.x + p3.z) * p3.y));
        }
        
        // Simplex-like noise for turbulence
        fn noise2D(p: vec2f) -> f32 {
            let i = floor(p);
            let f = fract(p);
            let u = f * f * (3.0 - 2.0 * f);
            return mix(
                mix(hash21(i + vec2f(0.0, 0.0)), hash21(i + vec2f(1.0, 0.0)), u.x),
                mix(hash21(i + vec2f(0.0, 1.0)), hash21(i + vec2f(1.0, 1.0)), u.x),
                u.y
            );
        }
        
        // Fractal Brownian Motion for realistic smoke turbulence
        fn fbm(p: vec2f, octaves: i32) -> f32 {
            var value = 0.0;
            var amplitude = 0.5;
            var frequency = 1.0;
            var pos = p;
            for (var i = 0; i < octaves; i++) {
                value += amplitude * noise2D(pos * frequency);
                amplitude *= 0.5;
                frequency *= 2.0;
            }
            return value;
        }

        // Realistic smoke pattern with turbulence
        fn smokePattern(localPos: vec2f, size: f32, progress: f32, seed: f32) -> f32 {
            let expandedSize = size * (0.5 + progress * 1.8);
            
            // Stretch along Y axis (shooting direction)
            var stretchedPos = localPos;
            stretchedPos.x *= 1.8;
            stretchedPos.y *= 0.6;
            
            // Offset smoke forward
            stretchedPos.y += expandedSize * 0.5;
            
            // Add turbulence that increases with progress
            let turbulenceScale = 3.0 + progress * 2.0;
            let turbulence = fbm(stretchedPos * turbulenceScale + vec2f(seed * 10.0, progress * 5.0), 4);
            stretchedPos += (turbulence - 0.5) * 0.3 * progress;
            
            let dist = length(stretchedPos);
            
            var density = 0.0;
            
            // Main smoke cloud with soft edges
            if (dist < expandedSize) {
                density = 1.0 - smoothstep(0.0, expandedSize, dist);
                // Add noise to edges for more natural look
                let edgeNoise = fbm(stretchedPos * 5.0 + vec2f(seed * 20.0), 3);
                density *= 0.7 + edgeNoise * 0.6;
            }
            
            // Multiple smoke puffs with varying sizes
            let puffCount = 6u;
            for (var i = 0u; i < puffCount; i++) {
                let fi = f32(i);
                let puffSeed = hash21(vec2f(seed * 100.0 + fi, fi * 3.7));
                let angle = fi * 6.28318 / f32(puffCount) + puffSeed * 1.5 - 1.57;
                let puffDist = expandedSize * 0.35 * (0.4 + progress * 0.6) * (0.7 + puffSeed * 0.6);
                let puffOffset = vec2f(cos(angle) * 0.6, sin(angle)) * puffDist;
                let puffSize = expandedSize * (0.25 + puffSeed * 0.2);
                
                let puffDistFromCenter = length(stretchedPos - puffOffset);
                
                if (puffDistFromCenter < puffSize) {
                    let puffDensity = (1.0 - smoothstep(0.0, puffSize, puffDistFromCenter)) * 0.4;
                    density = max(density, puffDensity);
                }
            }
            
            // Fade out with easing
            let fadeProgress = min(progress * 1.2, 1.0);
            density *= 1.0 - fadeProgress * fadeProgress * fadeProgress;
            
            return clamp(density, 0.0, 1.0);
        }

        // Realistic muzzle flash with irregular shape and color gradient
        fn flashPattern(localPos: vec2f, size: f32, progress: f32, seed: f32) -> vec4f {
            // Flash only visible in first 20% of animation (faster flash)
            if (progress > 0.2) {
                return vec4f(0.0);
            }
            
            let flashProgress = progress / 0.2;
            let flashSize = size * 1.3 * (1.0 - flashProgress * 0.4);
            
            // Offset flash to barrel tip
            var offsetPos = localPos;
            offsetPos.y += size * 0.25;
            
            let dist = length(offsetPos);
            let angle = atan2(offsetPos.x, -offsetPos.y);
            
            var intensity = 0.0;
            var coreIntensity = 0.0;
            
            // Irregular star pattern with 5-9 spikes based on seed
            let spikeCount = 5.0 + floor(hash21(vec2f(seed, 0.5)) * 4.0);
            let starAngle = angle * spikeCount;
            
            // Multiple layers of spikes for more complex shape
            var totalSpikePattern = 0.0;
            for (var layer = 0u; layer < 3u; layer++) {
                let layerF = f32(layer);
                let layerSeed = hash21(vec2f(seed * 50.0 + layerF, layerF * 2.3));
                let layerAngle = starAngle + layerSeed * 3.14159;
                
                let spikeIndex = floor((angle + 3.14159) / (6.28318 / spikeCount));
                let randomVariation = hash21(vec2f(seed * 100.0 + spikeIndex + layerF * 10.0, spikeIndex * 7.0 + layerF));
                let spikeMultiplier = 0.4 + randomVariation * 1.0;
                
                // Sharp irregular spikes
                let spike = pow(max(cos(layerAngle), 0.0), 1.2 + layerF * 0.5);
                totalSpikePattern += spike * spikeMultiplier * (1.0 - layerF * 0.25);
            }
            totalSpikePattern /= 2.0;
            
            let baseRadius = flashSize * 0.2;
            let spikeRadius = flashSize * 0.7;
            let starRadius = baseRadius + totalSpikePattern * spikeRadius;
            
            // Add noise to edges
            let edgeNoise = hash21(vec2f(angle * 10.0 + seed * 30.0, dist * 5.0));
            let noisyRadius = starRadius * (0.85 + edgeNoise * 0.3);
            
            if (dist < noisyRadius) {
                intensity = 1.0 - smoothstep(0.0, noisyRadius * 0.8, dist);
            }
            
            // Hot core
            let coreSize = flashSize * 0.3;
            if (dist < coreSize) {
                coreIntensity = 1.0 - smoothstep(0.0, coreSize, dist);
                coreIntensity = pow(coreIntensity, 0.7);
            }
            
            // Secondary glow around flash
            let glowSize = flashSize * 1.5;
            var glowIntensity = 0.0;
            if (dist < glowSize) {
                glowIntensity = (1.0 - smoothstep(0.0, glowSize, dist)) * 0.3;
            }
            
            // Combine intensities
            intensity = max(intensity, glowIntensity);
            
            // Fade out
            let fade = 1.0 - flashProgress * flashProgress;
            intensity *= fade;
            coreIntensity *= fade;
            
            // Color gradient: white core -> yellow -> orange at edges
            var color = vec3f(1.0, 0.85, 0.4); // Base orange-yellow
            color = mix(color, vec3f(1.0, 0.95, 0.8), intensity * 0.5); // Brighter toward center
            color = mix(color, vec3f(1.0, 1.0, 1.0), coreIntensity); // White hot core
            
            // Add slight red tint at very edges
            let edgeFactor = 1.0 - intensity;
            color = mix(color, vec3f(1.0, 0.5, 0.2), edgeFactor * 0.3 * (1.0 - coreIntensity));
            
            return vec4f(color, max(intensity, coreIntensity));
        }
        
        // Sparks flying from muzzle
        fn sparksPattern(localPos: vec2f, size: f32, progress: f32, seed: f32) -> vec4f {
            // Sparks visible in first 40% of animation
            if (progress > 0.4) {
                return vec4f(0.0);
            }
            
            let sparkProgress = progress / 0.4;
            
            var offsetPos = localPos;
            offsetPos.y += size * 0.2;
            
            var totalIntensity = 0.0;
            var sparkColor = vec3f(1.0, 0.9, 0.6);
            
            let sparkCount = 12u;
            for (var i = 0u; i < sparkCount; i++) {
                let fi = f32(i);
                let sparkSeed = hash22(vec2f(seed * 200.0 + fi, fi * 5.3));
                
                // Sparks spread in a cone forward
                let spreadAngle = (sparkSeed.x - 0.5) * 1.2 - 1.57; // Bias forward (-Y direction)
                let sparkSpeed = 0.5 + sparkSeed.y * 1.5;
                let sparkDist = sparkSpeed * sparkProgress * size * 2.5;
                
                let sparkPos = vec2f(
                    cos(spreadAngle) * sparkDist,
                    sin(spreadAngle) * sparkDist
                );
                
                let distToSpark = length(offsetPos - sparkPos);
                let sparkSize = size * 0.08 * (1.0 - sparkProgress * 0.7);
                
                if (distToSpark < sparkSize) {
                    let sparkIntensity = (1.0 - smoothstep(0.0, sparkSize, distToSpark));
                    // Sparks fade individually based on their "lifetime"
                    let sparkFade = 1.0 - pow(sparkProgress, 0.5 + sparkSeed.y);
                    totalIntensity = max(totalIntensity, sparkIntensity * sparkFade);
                }
            }
            
            // Vary spark colors slightly
            sparkColor = mix(sparkColor, vec3f(1.0, 0.7, 0.3), hash21(vec2f(seed, totalIntensity)) * 0.4);
            
            return vec4f(sparkColor, totalIntensity);
        }

        @fragment
        fn fs_main(
            @location(0) @interpolate(flat) instance_index: u32,
            @location(1) local_position: vec2f,
        ) -> @location(0) vec4f {
            let flashInfo = uFlashData[instance_index];
            let size = 1.0;
            let progress = flashInfo.x;
            let seed = flashInfo.y;
            
            var finalColor = vec3f(0.0);
            var totalAlpha = 0.0;
            
            // Layer 1: Smoke (background)
            let smokeDensity = smokePattern(local_position, size, progress, seed);
            if (smokeDensity > 0.0) {
                // Smoke color varies from light gray to darker as it dissipates
                let baseGray = mix(0.8, 0.45, progress);
                // Slight warm tint from the flash
                let warmth = max(0.0, 1.0 - progress * 3.0) * 0.15;
                let smokeColor = vec3f(
                    baseGray + warmth,
                    baseGray * 0.95 + warmth * 0.5,
                    baseGray * 0.9
                );
                
                let alpha = smokeDensity * 0.45;
                finalColor = mix(finalColor, smokeColor, alpha);
                totalAlpha = max(totalAlpha, alpha);
            }
            
            // Layer 2: Sparks (middle layer)
            let sparks = sparksPattern(local_position, size, progress, seed);
            if (sparks.a > 0.0) {
                finalColor = mix(finalColor, sparks.rgb, sparks.a);
                totalAlpha = max(totalAlpha, sparks.a);
            }
            
            // Layer 3: Flash (foreground)
            let flash = flashPattern(local_position, size, progress, seed);
            if (flash.a > 0.0) {
                // Additive blending for bright flash
                finalColor = mix(finalColor, flash.rgb, flash.a);
                // Boost alpha for flash to make it pop
                totalAlpha = max(totalAlpha, flash.a);
            }
            
            if (totalAlpha < 0.01) {
                discard;
            }
            
            return vec4f(finalColor, totalAlpha);
        }
    `,
);
