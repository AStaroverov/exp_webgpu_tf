import { ShaderMeta } from '../../../../../../../../src/WGSL/ShaderMeta.ts';
import { VariableKind, VariableMeta } from '../../../../../../../../src/Struct/VariableMeta.ts';
import { wgsl } from '../../../../../../../../src/WGSL/wgsl.ts';

export const shaderMeta = new ShaderMeta(
    {
        screenSize: new VariableMeta('uScreenSize', VariableKind.Uniform, `vec2<f32>`),
        time: new VariableMeta('uTime', VariableKind.Uniform, `f32`),

        // Tile size is now in physical pixels
        tileSize: new VariableMeta('uTileSize', VariableKind.Uniform, `f32`),
        pixelSize: new VariableMeta('uPixelSize', VariableKind.Uniform, `f32`),
        colorCount: new VariableMeta('uColorCount', VariableKind.Uniform, `f32`),
        grassDensity: new VariableMeta('uGrassDensity', VariableKind.Uniform, `f32`),
        windStrength: new VariableMeta('uWindStrength', VariableKind.Uniform, `f32`),
        windDirection: new VariableMeta('uWindDirection', VariableKind.Uniform, `vec2<f32>`),
    },
    {},
    // language=WGSL
    wgsl`        
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
                0.0,
                1.0
            );

            return VertexOutput(
                rect_vertex,
            );
        }
        
        // Improved hash function for better pseudo-random generation
        fn hash22(p: vec2f) -> vec2f {
            var p3 = fract(vec3f(p.xyx) * vec3f(0.1031, 0.1030, 0.0973));
            p3 += dot(p3, p3.yzx + 33.33);
            return fract((p3.xx + p3.yz) * p3.zy);
        }
        
        // Additional hash function for more variation
        fn hash21(p: vec2f) -> f32 {
            var p3 = fract(vec3f(p.xyx) * vec3f(0.1031, 0.1030, 0.0973));
            p3 += dot(p3, p3.yzx + 19.19);
            return fract((p3.x + p3.y) * p3.z);
        }
        
        // Noise function for wind variation
        fn noise(p: vec2f) -> f32 {
            let i = floor(p);
            let f = fract(p);
            
            let a = hash21(i);
            let b = hash21(i + vec2f(1.0, 0.0));
            let c = hash21(i + vec2f(0.0, 1.0));
            let d = hash21(i + vec2f(1.0, 1.0));
            
            let u = f * f * (3.0 - 2.0 * f);
            
            return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
        }
        
        // Enhanced wind effect with spatial and temporal variation
        fn calculatePixelWind(basePos: vec2f, height: f32, time: f32, grassType: f32) -> vec2f {
            // Use different frequencies based on position for natural wave patterns
            let large_scale = noise(basePos * 0.05 + time * 0.1);
            let medium_scale = noise(basePos * 0.1 + vec2f(time * 0.2, time * 0.15));
            let small_scale = noise(basePos * 0.3 + vec2f(sin(time * 0.3), cos(time * 0.27)));
            
            // Different grass types have different wind responses
            var windResponse = 1.0;
            if (grassType < 1.0) {
                windResponse = 1.2; // Taller grass moves more
            } else if (grassType < 2.0) {
                windResponse = 0.8; // L-shape moves less
            } else {
                windResponse = 1.0 + small_scale * 0.4; // Clusters have variable response
            }
            
            // Randomize phase based on position
            let phase_offset = hash21(floor(basePos * 0.2)) * 6.28;
            
            // Create multi-layered wind effect
            let mainWind = (sin(time * 1.0 + basePos.x * 0.2 + phase_offset) * 0.5 + 0.5) * large_scale;
            let secondaryWind = (sin(time * 1.7 + basePos.y * 0.13 + phase_offset * 2.0) * 0.3) * medium_scale;
            let detailWind = small_scale * 0.2;
            
            // Combine different scales for more natural movement
            let windEffectX = (mainWind + secondaryWind + detailWind) * windResponse;
            
            // Vertical component is usually smaller
            let windEffectY = (medium_scale * 0.15 + small_scale * 0.1) * windResponse;
            
            // Apply wind strength and height factor
            let windFactor = uWindStrength * height * 2.0;
            
            // For pixel art, quantize the wind effect to distinct states
            // but with more possible states for less uniform appearance
            let quantLevels = 5.0;
            let quantX = floor(windEffectX * quantLevels) / quantLevels;
            let quantY = floor(windEffectY * quantLevels) / quantLevels;
            
            return vec2f(quantX, quantY) * windFactor;
        }
        
        @fragment
        fn fs_main(@builtin(position) fragCoord : vec4<f32>) -> @location(0) vec4f {
            let coord = vec2f(fragCoord.x, -fragCoord.y);
            
            // Instead of using the projection matrix for normalization,
            // use the screen size directly to get fixed-size tiles
            // uTileSize now represents the size in physical pixels
            
            // Calculate normalized coordinates 
            // This is in 0-1 range based on screen dimensions
            let normalizedCoord = coord / uScreenSize;
            
            // Convert to -1 to 1 range for easier tile placement calculation
            let centeredCoord = normalizedCoord * 2.0 - 1.0;
            
            // Calculate the tile coordinate
            // We divide by the tile size to get consistent physical pixel size
            let tilesPerScreen = uScreenSize / uTileSize;
            let tileUV = floor(normalizedCoord * tilesPerScreen);
            let tileLocalUV = fract(normalizedCoord * tilesPerScreen);
            
            // Generate random values for this tile
            let rnd = hash22(tileUV);
            
            // Add some variation to each tile to avoid repetitive patterns
            let tileVariation = hash21(tileUV + vec2f(123.45, 678.9));
            
            // Define pixel art palette based around #ACC178
            let baseColor = vec4f(0.0);
            let midGreen = vec4f(0.675, 0.757, 0.471, 0.8);       // #ACC178 - Main olive green
            let lightGreen = vec4f(0.792, 0.839, 0.592, 0.8);     // #CAD697 - Light olive
            let brightGreen = vec4f(0.878, 0.914, 0.741, 0.8);    // #E0E9BD - Very light olive
            
            // Determine if this tile has grass based on density
            if (rnd.x > uGrassDensity) {
                discard;
            }
            
            // Randomize grass properties with more variation
            // Use continuous random value but quantize for pixel art look
            let grassTypeRnd = floor(rnd.y * 3.0 + tileVariation);
            let grassTypeFloat = grassTypeRnd % 3.0; // Ensure we stay within 0-2 range
            
            // Varied height adds to natural look
            let heightVar = hash21(tileUV * 7.89) * 0.3;
            let grassHeight = 0.4 + floor((rnd.y + heightVar) * 4.0) / 4.0 * 0.6;
            
            // Initialize with ground color
            var color = baseColor;
            
            // Calculate wind offset with improved variation
            let windOffset = calculatePixelWind(
                tileUV + vec2f(tileVariation * 10.0), // Add large offset for variation between tiles
                grassHeight,
                uTime + tileVariation * 6.28, // Phase variation
                grassTypeFloat
            );
            
            // Apply wind direction
            let xOffset = windOffset.x * uWindDirection.x;
            let yOffset = windOffset.y * uWindDirection.y;
            
            // Draw grass shape based on type
            if (grassTypeFloat < 1.0) {
                // Single straight blade with variations
                let grassWidth = 0.1 + tileVariation * 0.05; // Varied width
                let grassBaseX = floor((rnd.x + tileVariation * 0.5) * 5.0) / 5.0; // Varied position
                
                // Simple rectangular blade
                if (tileLocalUV.x > grassBaseX && tileLocalUV.x < grassBaseX + grassWidth) {
                    if (tileLocalUV.y < grassHeight) {
                        // Create discrete bands of color
                        if (tileLocalUV.y < grassHeight * 0.3) {
                            color = midGreen;
                        } else if (tileLocalUV.y < grassHeight * 0.6) {
                            color = lightGreen;
                        } else {
                            color = brightGreen;
                        }
                        
                        // Apply wind by offsetting the top part
                        if (tileLocalUV.y > grassHeight * 0.4) {
                            // Calculate stronger wind offset for more visible movement
                            let windStrength = (tileLocalUV.y - grassHeight * 0.4) / (grassHeight * 0.6);
                            
                            // Apply non-linear factor for more natural bending
                            let bendFactor = windStrength * windStrength;
                            
                            let appliedXOffset = xOffset * bendFactor;
                            let appliedYOffset = yOffset * bendFactor;
                            
                            // Check if we should draw this pixel with offset (both x and y)
                            if (!(tileLocalUV.x > grassBaseX + appliedXOffset && 
                                  tileLocalUV.x < grassBaseX + grassWidth + appliedXOffset &&
                                  tileLocalUV.y < grassHeight + appliedYOffset)) {
                                color = baseColor; // If wind moved grass away, show ground
                            }
                        }
                    }
                }
            } else if (grassTypeFloat < 2.0) {
                // Blocky L-shape grass with variations
                let baseX = floor((rnd.x + tileVariation * 0.3) * 3.0) / 3.0 + 0.1;
                let width = 0.15 - tileVariation * 0.05;
                let tipHeight = grassHeight * (0.7 + tileVariation * 0.2);
                
                // Wind effect
                let bendFactor = tileVariation * 0.5 + 0.5; // Random bend responsiveness
                let tipXOffset = xOffset * bendFactor;
                let tipYOffset = yOffset * bendFactor * 0.5;
                
                // Stem
                if (tileLocalUV.x > baseX && tileLocalUV.x < baseX + width) {
                    if (tileLocalUV.y < tipHeight) {
                        color = midGreen;
                    }
                }
                
                // Tip with wind
                if (tileLocalUV.y > tipHeight - 0.1 && tileLocalUV.y < grassHeight) {
                    if (tileLocalUV.x > baseX + tipXOffset && 
                        tileLocalUV.x < baseX + width + tipXOffset + 0.1 &&
                        tileLocalUV.y < grassHeight + tipYOffset) {
                        color = lightGreen;
                    }
                }
            } else {
                // Pixelated cluster of grass with variations
                let cluster = floor((rnd + tileVariation) * 4.0) / 4.0; // Quantize with variation
                let baseX = cluster.x * 0.5 + 0.1;
                
                // Simple pixel blocks for grass
                for (var i = 0; i < 3; i++) {
                    // Vary block position and height
                    let blockX = baseX + f32(i) * (0.12 + tileVariation * 0.04);
                    let blockHeight = grassHeight - abs(f32(i) - 1.0) * (0.1 + tileVariation * 0.05);
                    
                    // Individual wind effect for each blade in the cluster
                    let blockVar = hash21(vec2f(f32(i) * 10.0, tileUV.y + f32(i)));
                    
                    // Calculate individual wind offset for each grass blade
                    let blockWindOffset = calculatePixelWind(
                        tileUV + vec2f(f32(i) * 3.0, blockVar * 5.0),
                        blockHeight,
                        uTime + blockVar * 3.14,
                        grassTypeFloat + f32(i) * 0.1
                    );
                    
                    // Apply wind direction
                    let blockXOffset = blockWindOffset.x * uWindDirection.x;
                    let blockYOffset = blockWindOffset.y * uWindDirection.y;
                    
                    // Non-linear bending for natural look
                    let heightPct = tileLocalUV.y / blockHeight;
                    let bendFactor = heightPct * heightPct; // Quadratic for natural bending
                    
                    // Apply wind with diagonal movement
                    if (tileLocalUV.x > blockX + blockXOffset * bendFactor && 
                        tileLocalUV.x < blockX + 0.1 + blockXOffset * bendFactor && 
                        tileLocalUV.y < blockHeight + blockYOffset * bendFactor) {
                        
                        // Blocky color gradient
                        if (tileLocalUV.y < blockHeight * 0.3) {
                            color = midGreen;
                        } else if (tileLocalUV.y < blockHeight * 0.7) {
                            color = lightGreen;
                        } else {
                            color = brightGreen;
                        }
                    }
                }
            }
    
            return color;
        }
    `,
);