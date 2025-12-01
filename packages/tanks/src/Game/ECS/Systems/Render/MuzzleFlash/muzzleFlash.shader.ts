import { ShaderMeta } from '../../../../../../../renderer/src/WGSL/ShaderMeta.ts';
import { VariableKind, VariableMeta } from '../../../../../../../renderer/src/Struct/VariableMeta.ts';
import { wgsl } from '../../../../../../../renderer/src/WGSL/wgsl.ts';

// Maximum number of muzzle flash particles we can render in one pass
export const MAX_MUZZLE_FLASH_COUNT = 64;

export const shaderMeta = new ShaderMeta(
    {
        screenSize: new VariableMeta('uScreenSize', VariableKind.Uniform, `vec2<f32>`),
        flashCount: new VariableMeta('uFlashCount', VariableKind.Uniform, `u32`),
        zIndex: new VariableMeta('uZIndex', VariableKind.Uniform, `f32`),
        // Array of flash data: x, y, size, progress (0-1)
        flashData: new VariableMeta('uFlashData', VariableKind.StorageRead, `array<vec4<f32>, ${MAX_MUZZLE_FLASH_COUNT}>`),
        flashRotation: new VariableMeta('uFlashRotation', VariableKind.StorageRead, `array<f32, ${MAX_MUZZLE_FLASH_COUNT}>`),
    },
    {},
    // language=WGSL
    wgsl/* wgsl */`
        struct VertexOutput {
            @builtin(position) position: vec4f,
        };

        @vertex
        fn vs_main(
            @builtin(vertex_index) vertex_index: u32,
        ) -> VertexOutput {
            let rect_vertex = vec4<f32>(
                select(-1.0, 1.0, vertex_index > 0u && vertex_index < 4u),
                select(-1.0, 1.0, vertex_index > 1u && vertex_index < 5u),
                uZIndex,
                1.0
            );

            return VertexOutput(
                rect_vertex,
            );
        }

        // Hash function for pseudo-random generation
        fn hash21(p: vec2f) -> f32 {
            var p3 = fract(vec3f(p.xyx) * vec3f(0.1031, 0.1030, 0.0973));
            p3 += dot(p3, p3.yzx + 19.19);
            return fract((p3.x + p3.y) * p3.z);
        }

        // Egg-shaped smoke pattern (elongated in shooting direction)
        fn smokePattern(localPos: vec2f, size: f32, progress: f32, seed: f32) -> f32 {
            let expandedSize = size * (0.6 + progress * 1.44);
            
            // Stretch along Y axis (shooting direction after rotation)
            // Make it egg-shaped: narrower at back, wider at front
            var stretchedPos = localPos;
            stretchedPos.x *= 2.0; // Compress horizontally
            stretchedPos.y *= 0.7; // Elongate vertically
            
            // Offset smoke forward in shooting direction (outside the barrel)
            stretchedPos.y += expandedSize * 0.6;
            
            let dist = length(stretchedPos);
            
            var density = 0.0;
            
            // Main egg shape with smooth falloff
            if (dist < expandedSize) {
                density = 1.0 - smoothstep(0.0, expandedSize, dist);
            }
            
            // Add some smaller puffs around for smoke effect
            let puffCount = 4u;
            for (var i = 0u; i < puffCount; i++) {
                let angle = f32(i) * 6.28318 / f32(puffCount) + seed * 6.28 - 1.57; // Bias toward front
                let puffDist = expandedSize * 0.4 * (0.3 + progress * 0.5);
                let puffOffset = vec2f(cos(angle) * 0.5, sin(angle)) * puffDist;
                let puffSize = expandedSize * 0.35;
                
                let puffDistFromCenter = length(stretchedPos - puffOffset);
                
                if (puffDistFromCenter < puffSize) {
                    density = max(density, (1.0 - smoothstep(0.0, puffSize, puffDistFromCenter)) * 0.5);
                }
            }
            
            // Faster fade out (30% faster)
            let fadeProgress = min(progress * 1.3, 1.0);
            density *= 1.0 - fadeProgress * fadeProgress;
            
            return density;
        }

        // 7-pointed star flash pattern with random spike variations
        fn flashPattern(localPos: vec2f, size: f32, progress: f32, seed: f32) -> f32 {
            // Flash only visible in first 25% of animation
            if (progress > 0.25) {
                return 0.0;
            }
            
            let flashProgress = progress / 0.25; // Normalize to 0-1 for flash duration
            let flashSize = size * 1.5 * (1.0 - flashProgress * 0.3);
            
            // Offset flash forward in shooting direction
            var offsetPos = localPos;
            offsetPos.y += size * 1.0;
            
            let dist = length(offsetPos);
            let angle = atan2(offsetPos.x, -offsetPos.y);
            
            var flash = 0.0;
            
            // 7-pointed star with random spike lengths
            let points = 7.0;
            let starAngle = angle * points;
            
            // Generate random variations for each spike using seed
            // Each spike gets a different random multiplier
            let spikeIndex = floor((angle + 3.14159) / (6.28318 / points));
            let randomVariation = hash21(vec2f(seed * 100.0 + spikeIndex, spikeIndex * 7.0));
            let spikeMultiplier = 0.6 + randomVariation * 0.8; // Range 0.6 to 1.4
            
            // Create sharp spikes
            let spikePattern = pow(max(cos(starAngle), 0.0), 1.5);
            let baseRadius = flashSize * 0.25;
            let spikeRadius = flashSize * 0.9 * spikeMultiplier;
            let starRadius = baseRadius + spikePattern * spikeRadius;
            
            if (dist < starRadius) {
                flash = 1.0 - smoothstep(0.0, starRadius * 0.7, dist);
            }
            
            // Bright core at center
            let coreSize = flashSize * 0.35;
            if (dist < coreSize) {
                flash = max(flash, 1.0 - smoothstep(0.0, coreSize, dist));
            }
            
            // Fade out flash
            flash *= 1.0 - flashProgress * flashProgress;
            
            return flash;
        }

        @fragment
        fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4f {
            let screenPos = vec2f(fragCoord.x, fragCoord.y);
            
            var totalAlpha = 0.0;
            var finalColor = vec3f(0.0);
            
            // Check each muzzle flash
            for (var i = 0u; i < uFlashCount; i++) {
                let flashInfo = uFlashData[i];
                let flashX = flashInfo.x;
                let flashY = flashInfo.y;
                let flashSize = flashInfo.z;
                let progress = flashInfo.w;
                let rotation = uFlashRotation[i];
                
                // Transform screen position relative to flash center
                let flashCenter = vec2f(flashX, flashY);
                
                // Early exit if too far from flash center
                let maxRadius = flashSize * 4.0;
                if (length(screenPos - flashCenter) > maxRadius) {
                    continue;
                }
                
                // Apply rotation around flash center (rotate to align with shooting direction)
                let relPos = screenPos - flashCenter;
                let cosR = cos(-rotation);
                let sinR = sin(-rotation);
                let localPos = vec2f(
                    relPos.x * cosR - relPos.y * sinR,
                    relPos.x * sinR + relPos.y * cosR
                );
                
                // Generate seed for this flash
                let flashSeed = hash21(vec2f(f32(i), 0.0));
                
                // Calculate smoke density
                let smokeDensity = smokePattern(localPos, flashSize, progress, flashSeed);
                
                // Calculate flash intensity
                let flashIntensity = flashPattern(localPos, flashSize, progress, flashSeed);
                
                // Combine smoke and flash
                if (flashIntensity > 0.0) {
                    // Flash color: pure bright white
                    let flashColor = vec3f(1.0, 1.0, 1.0);
                    
                    let alpha = flashIntensity;
                    finalColor = mix(finalColor, flashColor, alpha);
                    totalAlpha = max(totalAlpha, alpha);
                }
                
                if (smokeDensity > 0.0) {
                    // Smoke color: light gray, more transparent
                    let baseGray = mix(0.85, 0.5, progress);
                    let smokeColor = vec3f(baseGray, baseGray * 0.95, baseGray * 0.9);
                    
                    // More transparent smoke (reduced from 0.8 to 0.5)
                    let alpha = smokeDensity * 0.5;
                    finalColor = mix(finalColor, smokeColor, alpha);
                    totalAlpha = max(totalAlpha, alpha);
                }
            }
            
            if (totalAlpha < 0.01) {
                discard;
            }
            
            return vec4f(finalColor, totalAlpha);
        }
    `,
);
