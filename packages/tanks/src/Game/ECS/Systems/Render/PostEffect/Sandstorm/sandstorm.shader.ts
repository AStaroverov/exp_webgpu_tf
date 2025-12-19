import { ShaderMeta } from '../../../../../../../../renderer/src/WGSL/ShaderMeta.ts';
import { VariableKind, VariableMeta } from '../../../../../../../../renderer/src/Struct/VariableMeta.ts';
import { wgsl } from '../../../../../../../../renderer/src/WGSL/wgsl.ts';
import { noiseWGSL } from '../../noise.wgsl.ts';

export const shaderMeta = new ShaderMeta(
    {
        screenSize: new VariableMeta('uScreenSize', VariableKind.Uniform, `vec2<f32>`),
        time: new VariableMeta('uTime', VariableKind.Uniform, `f32`),
        windDirection: new VariableMeta('uWindDirection', VariableKind.Uniform, `vec2<f32>`),
        uMapOffset: new VariableMeta('uMapOffset', VariableKind.Uniform, `vec2<f32>`),
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
        fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
            let pos = array<vec2f, 6>(
                vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
                vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
            );
            let uvs = array<vec2f, 6>(
                vec2f(0.0, 1.0), vec2f(1.0, 1.0), vec2f(0.0, 0.0),
                vec2f(0.0, 0.0), vec2f(1.0, 1.0), vec2f(1.0, 0.0)
            );

            var out: VertexOutput;
            out.position = vec4f(pos[vertex_index], 1.0, 1.0); // z=1.0 for front in reverse depth
            out.uv = uvs[vertex_index];
            return out;
        }

        @fragment
        fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
            let pixelPos = uv * uScreenSize;
            
            // Saturated Mad Max colors
            let darkSand = vec3f(0.45, 0.22, 0.08);
            let midSand = vec3f(0.75, 0.38, 0.12);
            let lightSand = vec3f(0.9, 0.6, 0.3);
            
            // Large scale movement for massive clouds
            let slowTime = uTime * 0.2;
            let windDir = normalize(uWindDirection);
            
            // World-space UV for the storm - full offset compensation
            let worldUV = (pixelPos + uMapOffset) * 0.001;
            
            // Base layer: Huge rolling waves of dust
            let wave1 = fbm4(worldUV - windDir * slowTime * 1.5);
            let wave2 = fbm4(worldUV * 1.5 - windDir * slowTime * 2.0 + 10.0);
            
            // Modulation: periods of intense storm and relative calm
            let intensity = sin(uTime * 0.1 + worldUV.x * 0.5) * 0.5 + 0.5;
            
            var dustCloud = mix(wave1, wave2, 0.5);
            dustCloud = smoothstep(0.3, 0.8, dustCloud * (0.5 + intensity * 0.5));
            
            // Add some "shredded" look for the wind
            let detail = valueNoise(worldUV * 10.0 - windDir * uTime * 3.0);
            dustCloud += detail * 0.15 * dustCloud;

            // === SUPER STRONG GRAIN / FILM NOISE ===
            // Multiple layers of grain for intense texture
            
            // Fast-moving grain (per-pixel, time-based)
            let grainCoord1 = pixelPos + uTime * 1000.0;
            let grain1 = grainNoise(grainCoord1, 1.0);
            
            // Medium grain layer
            let grainCoord2 = pixelPos * 0.5 + uTime * 500.0 + 1234.0;
            let grain2 = grainNoise(grainCoord2, 0.8);
            
            // Slower, larger grain clumps
            let grainCoord3 = floor(pixelPos * 0.25) + floor(uTime * 60.0);
            let grain3 = grainNoise(grainCoord3, 0.6);
            
            // Combine grain layers
            let totalGrain = grain1 * 0.5 + grain2 * 0.3 + grain3 * 0.2;
            
            // Sand particle simulation - bright flying specks
            let particleCoord = floor(pixelPos * 0.3) + floor(uTime * 30.0) * windDir * 10.0;
            let particle = hash21(particleCoord);
            let sparkle = select(0.0, 0.4, particle > 0.97);

            // Final color mapping
            var finalColor = mix(midSand, lightSand, wave1);
            finalColor = mix(darkSand, finalColor, dustCloud);
            
            // Apply grain - stronger effect
            finalColor += vec3f(totalGrain * 0.25);
            
            // Add bright sand particles
            finalColor += vec3f(sparkle);

            // Opacity control: more intense clouds are more opaque
            let alpha = clamp(dustCloud * 0.7, 0.0, 0.85);

            if (alpha < 0.01) {
                discard;
            }

            return vec4f(finalColor, alpha);
        }
    `
);
