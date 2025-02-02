import { VariableKind, VariableMeta } from '../../../Struct/VariableMeta.ts';
import { ShaderMeta } from '../../../WGSL/ShaderMeta.ts';
import { wgsl } from '../../../WGSL/wgsl.ts';
import { ROPE_POINTS_COUNT, ROPE_SEGMENTS_COUNT } from '../../Component/Rope.ts';

export const MAX_ROPES_COUNT = 1000;
export const MAX_INSTANCE_COUNT = MAX_ROPES_COUNT * ROPE_POINTS_COUNT;

export const shaderMeta = new ShaderMeta(
    {
        resolution: new VariableMeta('uResolution', VariableKind.StorageRead, `array<vec2<f32>, ${ MAX_ROPES_COUNT }>`),
        translate: new VariableMeta('uTranslate', VariableKind.StorageRead, `array<vec2<f32>, ${ MAX_ROPES_COUNT }>`),

        thinness: new VariableMeta('uThinness', VariableKind.StorageRead, `array<f32, ${ MAX_ROPES_COUNT }>`),
        color: new VariableMeta('uColor', VariableKind.StorageRead, `array<vec4<f32>, ${ MAX_ROPES_COUNT }>`),
        points: new VariableMeta('uPoints', VariableKind.StorageRead, `array<vec2<f32>, ${ MAX_INSTANCE_COUNT }>`),
    },
    {},
    // language=WGSL
    wgsl`
        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) @interpolate(flat) instance_index: u32
        };

        @vertex
        fn vertex(
            @builtin(vertex_index) vertex_index : u32,
            @builtin(instance_index) instance_index : u32,
        ) -> VertexOutput {
            var rope_index = u32(instance_index / ${ROPE_SEGMENTS_COUNT});

             if (uPoints[instance_index + 1].x == 0.0 && uPoints[instance_index + 1].y == 0.0) {
                 return VertexOutput(vec4<f32>(0), instance_index);
             }

            var rectVertex = compute_rect_vertex_position(
                uPoints[instance_index],
                uPoints[instance_index + 1],
                vertex_index,
                uThinness[rope_index] + 1000,
            );
            var position = vec4<f32>(
                to_final_position(uResolution[rope_index], uTranslate[rope_index], rectVertex),
                0.0,
                1.0
            );
            
            return VertexOutput(position, instance_index);
        }
        
        @fragment
        fn fragment(
            @builtin(position) position: vec4<f32>,
            @location(0) @interpolate(flat) instance_index: u32,
        ) -> @location(0) vec4<f32> {
            if (position.w == 0) { discard; }
        
            var rope_index = u32(instance_index / ${ROPE_SEGMENTS_COUNT});

            // dist incorrect
            var dist = sd_segment(
                position.xy,
                uPoints[instance_index].xy,
                uPoints[instance_index + 1].xy,
                uThinness[rope_index] / 2.0
            );
            if (dist > 0.0) {discard;} 

            return uColor[rope_index];
        }
        
        const RECT_VERTEX_BASIS = array<vec2<f32>, 6>(
            vec2<f32>(0.0, 0.5),  // 0
            vec2<f32>(1.0, 0.5),  // 1
            vec2<f32>(0.0, -0.5), // 2
            vec2<f32>(1.0, 0.5),  // 3
            vec2<f32>(0.0, -0.5), // 4
            vec2<f32>(1.0, -0.5)  // 5
        );

        fn compute_rect_vertex_position(pointA: vec2<f32>, pointB: vec2<f32>, index: u32, width: f32) -> vec2<f32> {
            // what happen here - https://wwwtyro.net/2019/11/18/instanced-lines.html
            var basis = RECT_VERTEX_BASIS[index];
            var len = length(pointB - pointA);
            var xBasis = normalize(pointB - pointA);
            var yBasis = normalize(vec2<f32>(-xBasis.y, xBasis.x));

            return pointA
                + xBasis * len * basis.x // center of segment
                + xBasis * width * (basis.x - 0.5) // place for circle
                + yBasis * width * basis.y; // width of segment
        }
        
        fn to_final_position(res: vec2<f32>, tran: vec2<f32>, pos: vec2<f32>) -> vec2<f32> {
            return vec2<f32>(
                (((pos.x + tran.x) / res.x) * 2.0 - 1.0),
                -(((pos.y + tran.y) / res.y) * 2.0 - 1.0)
            );
        }

        fn sd_segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, th: f32) -> f32 {
            var pa = p - a;
            var ba = b - a;
            var h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
            return length(pa - ba * h) - th;
        }
    `,
);

// console.log('>>', shaderMeta.shader.trim());
