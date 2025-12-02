import { ShaderMeta } from '../../../../../../../renderer/src/WGSL/ShaderMeta.ts';
import { VariableKind, VariableMeta } from '../../../../../../../renderer/src/Struct/VariableMeta.ts';
import { wgsl } from '../../../../../../../renderer/src/WGSL/wgsl.ts';

// Maximum number of hit flash instances
export const MAX_HIT_FLASH_COUNT = 64;

export const shaderMeta = new ShaderMeta(
    {
        projection: new VariableMeta('uProjection', VariableKind.Uniform, `mat4x4<f32>`),
        // Array of flash transforms (position encoded in matrix)
        transform: new VariableMeta('uTransform', VariableKind.StorageRead, `array<mat4x4<f32>, ${MAX_HIT_FLASH_COUNT}>`),
        // Flash data: size, progress (0-1), seed, unused
        flashData: new VariableMeta('uFlashData', VariableKind.StorageRead, `array<vec4<f32>, ${MAX_HIT_FLASH_COUNT}>`),
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
            let size = flashInfo.x;
            let progress = flashInfo.y;
            
            // Calculate expanded size for sparks
            let expandedSize = size * (1.0 + progress * 0.5);
            let maxRadius = expandedSize * 2.5;
            
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

        // Radial spark pattern - sparks flying outward in all directions
        fn sparkPattern(localPos: vec2f, size: f32, progress: f32, seed: f32) -> f32 {
            // Scale size 2x for bigger effect
            let scaledSize = size * 2.0;
            let dist = length(localPos);
            let angle = atan2(localPos.y, localPos.x);
            
            // Sparks expand outward over time
            let expandSpeed = scaledSize * 2.0;
            let sparkTravel = progress * expandSpeed;
            
            var sparkIntensity = 0.0;
            
            // Multiple spark rays
            let sparkCount = 12u;
            for (var i = 0u; i < sparkCount; i++) {
                // Random angle for each spark
                let sparkSeed = hash21(vec2f(seed * 100.0 + f32(i), f32(i) * 17.0));
                let sparkAngle = f32(i) * 6.28318 / f32(sparkCount) + sparkSeed * 0.5;
                
                // Random length for each spark (2x longer)
                let sparkLength = scaledSize * (0.8 + hash21(vec2f(sparkSeed, seed)) * 1.2);
                
                // Spark position along its trajectory
                let sparkStartDist = scaledSize * 0.2;
                let sparkEndDist = sparkStartDist + sparkLength * progress;
                
                // Check if current pixel is on this spark's path
                let angleDiff = abs(atan2(sin(angle - sparkAngle), cos(angle - sparkAngle)));
                let sparkWidth = scaledSize * 0.12 * (1.0 - progress * 0.5); // Thicker sparks
                
                if (angleDiff < sparkWidth / max(dist, 0.1)) {
                    // Check if within spark's current extent
                    if (dist > sparkStartDist && dist < sparkEndDist) {
                        // Brightness falls off along the spark and over time (brighter)
                        let alongSpark = (dist - sparkStartDist) / (sparkEndDist - sparkStartDist);
                        let brightness = (1.0 - alongSpark * 0.7) * (1.0 - progress * 0.8);
                        sparkIntensity = max(sparkIntensity, brightness);
                    }
                }
            }
            
            return sparkIntensity;
        }

        // Central impact flash - bright core that fades quickly
        fn impactFlashPattern(localPos: vec2f, size: f32, progress: f32, seed: f32) -> f32 {
            // Scale size 2x for bigger effect
            let scaledSize = size * 2.0;
            
            // Flash only visible in first 40% of animation (longer duration)
            if (progress > 0.4) {
                return 0.0;
            }
            
            let flashProgress = progress / 0.4;
            let dist = length(localPos);
            
            // Expanding ring effect (2x bigger)
            let ringRadius = scaledSize * 0.4 * (1.0 + flashProgress * 2.0);
            let ringWidth = scaledSize * 0.5 * (1.0 - flashProgress * 0.5);
            
            var flash = 0.0;
            
            // Bright core (2x bigger)
            let coreSize = scaledSize * 0.6 * (1.0 - flashProgress * 0.6);
            if (dist < coreSize) {
                flash = 1.0 - smoothstep(0.0, coreSize, dist) * 0.5; // Brighter core
            }
            
            // Expanding ring (brighter)
            let ringDist = abs(dist - ringRadius);
            if (ringDist < ringWidth) {
                let ringIntensity = (1.0 - ringDist / ringWidth) * 0.9;
                flash = max(flash, ringIntensity);
            }
            
            // Slower fade out
            flash *= 1.0 - flashProgress * flashProgress * 0.7;
            
            return flash;
        }

        @fragment
        fn fs_main(
            @location(0) @interpolate(flat) instance_index: u32,
            @location(1) local_position: vec2f,
        ) -> @location(0) vec4f {
            let flashInfo = uFlashData[instance_index];
            let size = flashInfo.x;
            let progress = flashInfo.y;
            let seed = flashInfo.z;
            
            var totalAlpha = 0.0;
            var finalColor = vec3f(0.0);
            
            // Calculate spark intensity
            let sparkIntensity = sparkPattern(local_position, size, progress, seed);
            
            // Calculate impact flash intensity
            let flashIntensity = impactFlashPattern(local_position, size, progress, seed);
            
            // Combine effects
            if (flashIntensity > 0.0) {
                // Impact flash: bright yellow-white
                let flashColor = vec3f(1.0, 0.95, 0.8);
                
                let alpha = flashIntensity;
                finalColor = mix(finalColor, flashColor, alpha);
                totalAlpha = max(totalAlpha, alpha);
            }
            
            if (sparkIntensity > 0.0) {
                // Sparks: bright orange-yellow color
                let sparkColor = mix(
                    vec3f(1.0, 0.9, 0.5),  // Bright yellow
                    vec3f(1.0, 0.6, 0.2),  // Bright orange
                    progress
                );
                
                let alpha = min(sparkIntensity * 1.2, 1.0); // Brighter sparks
                finalColor = mix(finalColor, sparkColor, alpha);
                totalAlpha = max(totalAlpha, alpha);
            }
            
            if (totalAlpha < 0.01) {
                discard;
            }
            
            return vec4f(finalColor, totalAlpha);
        }
    `,
);
