import { VariableKind, VariableMeta } from '../../../Struct/VariableMeta.ts';
import { ShaderMeta } from '../../../WGSL/ShaderMeta.ts';
import { wgsl } from '../../../WGSL/wgsl.ts';

export const MAX_INSTANCE_COUNT = 100;

export const shaderMeta = new ShaderMeta(
    {
        resolution: new VariableMeta('uResolution', VariableKind.StorageRead, `array<vec2<f32>, ${ MAX_INSTANCE_COUNT }>`),
        translate: new VariableMeta('uTranslate', VariableKind.StorageRead, `array<vec2<f32>, ${ MAX_INSTANCE_COUNT }>`),

        // 0: circle, 1: segment, 2: rectangle, 3: rhombus
        kind: new VariableMeta('uKind', VariableKind.StorageRead, `array<u32, ${ MAX_INSTANCE_COUNT }>`),
        color: new VariableMeta('uColor', VariableKind.StorageRead, `array<vec4<f32>, ${ MAX_INSTANCE_COUNT }>`),
        point1: new VariableMeta('uPoint1', VariableKind.StorageRead, `array<vec2<f32>, ${ MAX_INSTANCE_COUNT }>`),
        point2: new VariableMeta('uPoint2', VariableKind.StorageRead, `array<vec2<f32>, ${ MAX_INSTANCE_COUNT }>`),
        thinness: new VariableMeta('uThinness', VariableKind.StorageRead, `array<f32, ${ MAX_INSTANCE_COUNT }>`),
        roundness: new VariableMeta('uRoundness', VariableKind.StorageRead, `array<f32, ${ MAX_INSTANCE_COUNT }>`),
    },
    {},
    // language=WGSL
    wgsl`
        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) @interpolate(flat) instance_index: u32
        };
        
        @vertex
        fn vs_main(
            @builtin(vertex_index) vertex_index: u32,
            @builtin(instance_index) instance_index: u32
        ) -> VertexOutput {
            var kind = uKind[instance_index];
            var thinness = uThinness[instance_index];
            var min = 0.0;
            var max = 0.0;
            var point1 = uPoint1[instance_index];
            var point2 = uPoint2[instance_index];
            
            // circle
            if (kind == 0u) {
                min = vec2<f32>(-thinness, -thinness);
                max = vec2<f32>(thinness, thinness);
            }
            // rhombus
            if (kind == 3u) {
                point2 = point1 + point2;
                point1 = point1 - point2;
            }
            
            let min = vec2(min(point1.x, point2.x), min(point1.y, point2.y)) - vec2(thinness);// - vec2(100.0, 100.0);
            let max = vec2(max(point1.x, point2.x), max(point1.y, point2.y)) + vec2(thinness);// + vec2(100.0, 100.0);
            let rect_vertex = vec2<f32>(
                select(min.x, max.x, vertex_index > 0u && vertex_index < 4u),
                select(min.y, max.y, vertex_index > 1u && vertex_index < 5u),
            );
        
            var position = vec4<f32>(
                to_final_position(uResolution[instance_index], uTranslate[instance_index], rect_vertex),
                0.0,
                1.0
            );
            
            return VertexOutput(position, instance_index);
        }
        
        @fragment
        fn fs_main(
            @builtin(position) frag_coord: vec4<f32>,
            @location(0) @interpolate(flat) instance_index: u32
        ) -> @location(0) vec4<f32> {
            var pos = frag_coord.xy;
            var kind = uKind[instance_index];
            var point1 = uPoint1[instance_index];
            var point2 = uPoint2[instance_index];
            var thinness = uThinness[instance_index];
            var roundness = uRoundness[instance_index];
            var dist = 1.0;
            
            if (kind == 0u) {
                dist = sd_circle(pos, point1, thinness);
            } else if (kind == 1u) {
                dist = sd_circle(pos, point1, thinness); // replace with something
            } else if (kind == 2u) {
                dist = sd_rectangle(pos, point1, point2, thinness - roundness);
            } else if (kind == 3u) {
                dist = sd_rhombus(pos, point1, point2 - roundness);
            } else if (kind == 4u) {
                dist = sd_parallelogram(pos, point2.x - point1.x, point2.y - point1.y, point1.x - point2.x);
            }
            
            if (kind != 0u) {
                dist = op_round(dist, roundness);
            }
        
            if (dist > 0.0) { discard; }
        
            return uColor[instance_index];
        }
        
        fn to_final_position(res: vec2<f32>, tran: vec2<f32>, pos: vec2<f32>) -> vec2<f32> {
            return vec2<f32>(
                (((pos.x + tran.x) / res.x) * 2.0 - 1.0),
                -(((pos.y + tran.y) / res.y) * 2.0 - 1.0)
            );
        }
        
        fn op_round(d: f32, r: f32) -> f32 {
          return d - r;
        }
        
//        fn sd_simple_shape_dist(
//            frag_pos: vec2<f32>,
//            kind: u32,
//            point1: vec2<f32>,
//            point2: vec2<f32>,
//            thinness: f32,
//        ) -> f32 {
//            if (kind == 0u) {
//                return sd_circle(frag_pos, point1, thinness);
//            } else if (kind == 1u) {
//                return sd_segment(frag_pos, point1, point2, thinness);
//            } else if (kind == 2u) {
//                return sd_rectangle(frag_pos, point1, point2, thinness);
//            } else if (kind == 3u) {
////                return sd_circle(frag_pos, point1, point2.x);
//                return sd_rhombus(frag_pos, point1, point2);
//            }
//        
//            return 1.0;
//        }
        
        fn sd_circle(p: vec2<f32>, c: vec2<f32>, th: f32) -> f32 {
            return length(p - c) - th;
        }
        
        fn sd_segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, th: f32) -> f32 {
            var pa = p - a;
            var ba = b - a;
            var h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
            return length(pa - ba * h) - th;
        }
        
        fn sd_rectangle(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, th: f32) -> f32 {
            var l = length(b-a);
            var d = (b-a)/l;
            var q = p-(a+b)*0.5;
                q = mat2x2(d.x,-d.y,d.y,d.x)*q;
                q = abs(q)-vec2(l*0.5,th);
        
            return length(max(q, vec2(0.0))) + min(max(q.x,q.y),0.0);
        }
        
        fn ndot(a: vec2<f32>, b: vec2<f32>) -> f32 {
            return a.x*b.x - a.y*b.y;
        } 
        fn sd_rhombus(_p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
            var p = abs(_p - a);
            var h = clamp( ndot(b - 2.0*p,b)/dot(b,b), -1.0, 1.0 );
            var d = length( p - 0.5*b*vec2(1.0-h,1.0+h) );
            return d * sign( p.x*b.y + p.y*b.x - b.x*b.y );
        }
        
        fn sd_parallelogram(p: vec2<f32>, wi: f32, he: f32, sk: f32) -> f32 {
            let e = vec2<f32>(sk, he);
            let e2 = sk * sk + he * he;
            var pos = select(p, -p, p.y < 0.0);
            // horizontal edge
            var w = pos - e;
            w.x = w.x - clamp(w.x, -wi, wi);
            var d = vec2<f32>(dot(w, w), -w.y);
            // vertical edge
            let s = pos.x * e.y - pos.y * e.x;
            pos = select(pos, -pos, s < 0.0);
            var v = pos - vec2<f32>(wi, 0.0);
            v = v - e * clamp(dot(v, e) / e2, -1.0, 1.0);
            d = min(d, vec2<f32>(dot(v, v), wi * he - abs(s)));
            return sqrt(d.x) * sign(-d.y);
        }


        
        //fn sd_triangle(p: vec2<f32>, p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>) -> f32 {
        //    var e0 = p1-p0;
        //    var e1 = p2-p1;
        //    var e2 = p0-p2;
        //    var v0 = p -p0;
        //    var v1 = p -p1;
        //    var v2 = p -p2;
        //    var pq0 = v0 - e0*clamp( dot(v0,e0)/dot(e0,e0), 0.0, 1.0 );
        //    var pq1 = v1 - e1*clamp( dot(v1,e1)/dot(e1,e1), 0.0, 1.0 );
        //    var pq2 = v2 - e2*clamp( dot(v2,e2)/dot(e2,e2), 0.0, 1.0 );
        //    var s = sign( e0.x*e2.y - e0.y*e2.x );
        //    var d = min(min(vec2(dot(pq0,pq0), s*(v0.x*e0.y-v0.y*e0.x)),
        //                     vec2(dot(pq1,pq1), s*(v1.x*e1.y-v1.y*e1.x))),
        //                     vec2(dot(pq2,pq2), s*(v2.x*e2.y-v2.y*e2.x)));
        //    return -sqrt(d.x)*sign(d.y);
        //}
    `,
);

console.log('>>', shaderMeta.shader.trim());
