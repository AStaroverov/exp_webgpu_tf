import { ShaderMeta } from '../../../../../../../renderer/src/WGSL/ShaderMeta.ts';
import { VariableKind, VariableMeta } from '../../../../../../../renderer/src/Struct/VariableMeta.ts';
import { wgsl } from '../../../../../../../renderer/src/WGSL/wgsl.ts';
import { noiseWGSL } from '../noise.wgsl.ts';

export const shaderMeta = new ShaderMeta(
    {
        screenSize: new VariableMeta('uScreenSize', VariableKind.Uniform, `vec2<f32>`),
        time: new VariableMeta('uTime', VariableKind.Uniform, `f32`),
        mapOffset: new VariableMeta('uMapOffset', VariableKind.Uniform, `vec2<f32>`),
        tileSize: new VariableMeta('uTileSize', VariableKind.Uniform, `f32`),
        pixelSize: new VariableMeta('uPixelSize', VariableKind.Uniform, `f32`),
        density: new VariableMeta('uDensity', VariableKind.Uniform, `f32`),
        windStrength: new VariableMeta('uWindStrength', VariableKind.Uniform, `f32`),
        windDirection: new VariableMeta('uWindDirection', VariableKind.Uniform, `vec2<f32>`),
    },
    {},
    // language=WGSL
    wgsl`
        ${noiseWGSL}
        
        struct VertexOutput {
            @builtin(position) position: vec4f,
            @location(0) uv: vec2f,
        };
        
        @vertex
        fn vs_main(
            @builtin(vertex_index) vertex_index: u32,
        ) -> VertexOutput {
            let pos = array<vec2f, 6>(
                vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
                vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
            );
            let uvs = array<vec2f, 6>(
                vec2f(0.0, 1.0), vec2f(1.0, 1.0), vec2f(0.0, 0.0),
                vec2f(0.0, 0.0), vec2f(1.0, 1.0), vec2f(1.0, 0.0)
            );

            var out: VertexOutput;
            out.position = vec4f(pos[vertex_index], 0.0, 1.0);
            out.uv = uvs[vertex_index];
            return out;
        }

        @fragment
        fn fs_main(@builtin(position) fragCoord: vec4<f32>, @location(0) uv: vec2f) -> @location(0) vec4f {
            let coord = vec2f(fragCoord.x, -fragCoord.y);
            
            // World-space coordinates - pixel scale for crisp texture
            let worldPos = coord + vec2f(uMapOffset.x, -uMapOffset.y);
            
            // === DESERT PALETTE - Warm saturated tones ===
            let darkBrown = vec3f(0.58, 0.38, 0.22);      // Dark earthy brown
            let midBrown = vec3f(0.76, 0.55, 0.32);       // Mid warm brown
            let sandYellow = vec3f(0.88, 0.70, 0.42);     // Main sand color
            let lightSand = vec3f(0.94, 0.80, 0.52);      // Light sand
            let paleSand = vec3f(0.96, 0.86, 0.62);       // Pale highlights
            
            // === MULTI-SCALE NOISE FOR REALISTIC SAND ===
            
            // Large scale variation (big dune areas)
            let largeScale = worldPos * 0.001;
            let duneZone = fbm4(largeScale) * 0.2;
            
            // Medium patches of color variation
            let medScale = worldPos * 0.005;
            let patches = fbm4(medScale + 100.0) * 0.3;
            
            // Small rocky/gravelly texture
            let smallScale = worldPos * 0.025;
            let gravel = valueNoise(smallScale) * 0.25;
            
            // Fine grain detail
            let fineScale = worldPos * 0.15;
            let fineGrain = hash21(floor(fineScale)) * 0.12;
            
            // Pixel-level dithering for that pixel-art feel
            let pixelNoise = hash21(floor(worldPos * 0.08)) * 0.06;
            
            // Combine all layers
            var sandValue = 0.45 + duneZone + patches + gravel + fineGrain + pixelNoise;
            sandValue = clamp(sandValue, 0.0, 1.0);
            
            // Smooth gradient mapping with 5 color stops
            var sandColor: vec3f;
            if (sandValue < 0.3) {
                sandColor = mix(darkBrown, midBrown, sandValue / 0.3);
            } else if (sandValue < 0.5) {
                sandColor = mix(midBrown, sandYellow, (sandValue - 0.3) / 0.2);
            } else if (sandValue < 0.7) {
                sandColor = mix(sandYellow, lightSand, (sandValue - 0.5) / 0.2);
            } else {
                sandColor = mix(lightSand, paleSand, (sandValue - 0.7) / 0.3);
            }
            
            // Add subtle color variation (slight hue shifts)
            let hueShift = valueNoise(worldPos * 0.02 + 500.0);
            sandColor.r += (hueShift - 0.5) * 0.04;
            sandColor.g += (hueShift - 0.5) * 0.02;
   
            // Occasional lighter spots (pebbles catching light)
            let pebbleNoise = hash21(floor(worldPos * 0.15 + 700.0));
            if (pebbleNoise > 0.94) {
                sandColor = mix(sandColor, paleSand, 0.3);
            }
            
            return vec4f(sandColor, 1.0);
        }
    `,
);
