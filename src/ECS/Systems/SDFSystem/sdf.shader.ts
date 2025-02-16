import { VariableKind, VariableMeta } from '../../../Struct/VariableMeta.ts';
import { ShaderMeta } from '../../../WGSL/ShaderMeta.ts';
import { wgsl } from '../../../WGSL/wgsl.ts';

export const MAX_INSTANCE_COUNT = 10000;

export const shaderMeta = new ShaderMeta(
    {
        projection: new VariableMeta('uProjection', VariableKind.Uniform, `mat4x4<f32>`),
        transform: new VariableMeta('uTransform', VariableKind.StorageRead, `array<mat4x4<f32>, ${ MAX_INSTANCE_COUNT }>`),

        // 0: circle, 1: rectangle, 2: rhombus
        kind: new VariableMeta('uKind', VariableKind.StorageRead, `array<u32, ${ MAX_INSTANCE_COUNT }>`),
        color: new VariableMeta('uColor', VariableKind.StorageRead, `array<vec4<f32>, ${ MAX_INSTANCE_COUNT }>`),
        values: new VariableMeta('uValues', VariableKind.StorageRead, `array<f32, ${ MAX_INSTANCE_COUNT * 6 }>`),
        roundness: new VariableMeta('uRoundness', VariableKind.StorageRead, `array<f32, ${ MAX_INSTANCE_COUNT }>`),

        // shadow [shadowFadeStart, shadowFadeEnd]
        shadow: new VariableMeta('uShadow', VariableKind.StorageRead, `array<vec2<f32>, ${ MAX_INSTANCE_COUNT }>`),
    },
    {},
    // language=WGSL
    wgsl`
        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) @interpolate(flat) instance_index: u32,
            @location(1) local_position: vec2<f32>,
        };
        
        @vertex
        fn vs_main(
            @builtin(vertex_index) vertex_index: u32,
            @builtin(instance_index) instance_index: u32
        ) -> VertexOutput {
            let rect_vertex = compute_rect_vertex(vertex_index, instance_index);
        
            let position = vec4<f32>(
                to_final_position(uTransform[instance_index], rect_vertex),
                0.0,
                1.0
            );
            
            return VertexOutput(position, instance_index, rect_vertex);
        }
        
        @fragment
        fn fs_main(
            @location(0) @interpolate(flat) instance_index: u32,
            @location(1) local_position: vec2<f32>,
        ) -> @location(0) vec4<f32> {
            var dist = sd_shape(local_position, instance_index);
            var color = uColor[instance_index];
            
            if (dist > 0.0 || color.a == 0.0) { 
                discard;
//                return vec4<f32>(1.0, 1.0, 1.0, 0.1);
            }

            return color;
        }

        @vertex
        fn vs_shadow(
            @builtin(vertex_index) vertex_index: u32,
            @builtin(instance_index) instance_index: u32
        ) -> VertexOutput {
            let fadeEnd = uShadow[instance_index][1]; 
            let original_vertex = compute_rect_vertex(vertex_index, instance_index);
            let rect_vertex = original_vertex + normalize(original_vertex) * fadeEnd;
        
            let position = vec4<f32>(
                to_final_position(uTransform[instance_index], rect_vertex),
                0.0,
                1.0
            );
            
            return VertexOutput(position, instance_index, rect_vertex);
        }

        @fragment
        fn fs_shadow(
            @location(0) @interpolate(flat) instance_index: u32,
            @location(1) local_position: vec2<f32>,
        ) -> @location(0) vec4<f32> {
            let fadeStart = uShadow[instance_index][0];
            let fadeEnd = uShadow[instance_index][1]; 
           
            if (fadeEnd == 0) {
                discard;
            }
            
            let dist = sd_shape(local_position, instance_index);
        
            if (dist <= 0.0 ) {
                discard;
            }

            let transform = uTransform[instance_index];
            let rotation = mat2x2<f32>(transform[0].xy, transform[1].xy);
            let light_dir = normalize(vec2<f32>(-0.5, -0.5) * rotation);

            let shadow = compute_shadow(local_position, light_dir, instance_index);
//            return vec4<f32>(vec3<f32>(0.0), shadow);
                            
            let brightnessFactor = 1.0 - smoothstep(fadeStart, fadeEnd, abs(dist));
        
            return vec4<f32>(uColor[instance_index].rgb * 0.2, shadow * brightnessFactor);
        }
        
        fn compute_rect_vertex(vertex_index: u32, instance_index: u32) -> vec2<f32> {
            var kind = uKind[instance_index];
            var width = uValues[instance_index*6+0];
            var height = uValues[instance_index*6+1];
            
            // circle
            if (kind == 0u) {
                height = width;
            }
            
            if (kind == 3u) {
                width += abs(uValues[instance_index*6+2]) * 2.0;
            }
            
            if (kind == 5u) {
                width = max(width, max(uValues[instance_index*6+2], uValues[instance_index*6+4])) * 2;
                height = max(height, max(uValues[instance_index*6+3], uValues[instance_index*6+5])) * 2;
            }

            let min = vec2<f32>(-width/2, -height/2);
            let max = vec2<f32>( width/2,  height/2);
            
            let rect_vertex = vec2<f32>(
                select(min.x, max.x, vertex_index > 0u && vertex_index < 4u),
                select(min.y, max.y, vertex_index > 1u && vertex_index < 5u),
            );
            
            return rect_vertex;
        }

        fn to_final_position(transform: mat4x4<f32>, pos: vec2<f32>) -> vec2<f32> {
            var res = (uProjection * transform * vec4<f32>(pos, 0.0, 1.0)).xy; 
            return vec2<f32>(res.x, -res.y);
        }
            
        fn op_round(d: f32, r: f32) -> f32 {
          return d - r;
        }
        
        fn sd_shape(pos: vec2<f32>, instance_index: u32) -> f32 {
            var kind = uKind[instance_index];
            var width = uValues[instance_index*6+0];
            var height = uValues[instance_index*6+1];
            var roundness = uRoundness[instance_index];
            var dist = 1.0;
 
            if (kind == 0u) {
                dist = sd_circle(pos, width / 2);
            } else if (kind == 1u) {
                dist = sd_rectangle(pos, width / 2 - roundness, height / 2 - roundness);
            } else if (kind == 2u) {
                dist = sd_rhombus(pos, width / 2 - roundness, height / 2 - roundness);
            } else if (kind == 3u) {
                dist = sd_parallelogram(
                    pos,
                    width / 2 - roundness,
                    height / 2 - roundness,
                    uValues[instance_index*6+2]
                );
            } else if (kind == 4u) {
                dist = sd_trapezoid(
                    pos,
                    width / 2 - roundness,
                    uValues[instance_index*6+2] / 2 - roundness,
                    height / 2 - roundness
                );
            } else if (kind == 5u) {
                var ax = uValues[instance_index*6+0] - sign(uValues[instance_index*6+0])*roundness;
                var ay = uValues[instance_index*6+1] - sign(uValues[instance_index*6+1])*roundness;
                var bx = uValues[instance_index*6+2] - sign(uValues[instance_index*6+2])*roundness;
                var by = uValues[instance_index*6+3] - sign(uValues[instance_index*6+3])*roundness;
                var cx = uValues[instance_index*6+4] - sign(uValues[instance_index*6+4])*roundness;
                var cy = uValues[instance_index*6+5] - sign(uValues[instance_index*6+5])*roundness;
                dist = sd_triangle(pos, vec2f(ax, ay), vec2f(bx, by), vec2f(cx, cy));
            }
            
            if (kind != 0u) {
                dist = op_round(dist, roundness);
            }
            
            return dist;
        }
        
        fn sd_circle(p: vec2<f32>, r: f32) -> f32 {
            return length(p) - r;
        }
        
        fn sd_segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, th: f32) -> f32 {
            var pa = p - a;
            var ba = b - a;
            var h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
            return length(pa - ba * h) - th;
        }
        
        fn sd_rectangle(p: vec2<f32>, w: f32, h: f32) -> f32 {
            var b = vec2(w, h);
            var d = abs(p)-b;
            return length(max(d, vec2(0.0))) + min(max(d.x,d.y),(0.0));
        }
        
        fn ndot(a: vec2<f32>, b: vec2<f32>) -> f32 {
            return a.x*b.x - a.y*b.y;
        } 
        
        fn sd_rhombus(_p: vec2<f32>, wi: f32, he: f32) -> f32 {
            var p = abs(_p);
            var b = vec2<f32>(wi, he);
            var h = clamp( ndot(b - 2.0*p,b)/dot(b,b), -1.0, 1.0 );
            var d = length( p - 0.5*b*vec2(1.0-h,1.0+h) );
            return d * sign( p.x*b.y + p.y*b.x - b.x*b.y );
        }
        
        fn dot2(v: vec2<f32>) -> f32 {
            return dot(v, v);
        }
        
        fn sd_trapezoid(p: vec2<f32>, r1: f32, r2: f32, he: f32) -> f32 {
            let k1 = vec2<f32>(r2, he);
            let k2 = vec2<f32>(r2 - r1, 2.0 * he);
            var pp = p;
            pp.x = abs(pp.x);
            let selected_r = select(r2, r1, p.y < 0.0);
            let ca = vec2<f32>(
                max(0.0, pp.x - selected_r),
                abs(pp.y) - he
            );
            let cb = pp - k1 + k2 * clamp(dot(k1 - pp, k2) / dot2(k2), 0.0, 1.0);
            let s: f32 = select(1.0, -1.0, (cb.x < 0.0) && (ca.y < 0.0));
            return s * sqrt(min(dot2(ca), dot2(cb)));
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
        
        fn sd_triangle(p: vec2<f32>, p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>) -> f32 {
            let e0 = p1 - p0;
            let e1 = p2 - p1;
            let e2 = p0 - p2;
           
            let v0 = p - p0;
            let v1 = p - p1;
            let v2 = p - p2;
            
            let pq0 = v0 - e0 * clamp(dot(v0, e0) / dot(e0, e0), 0.0, 1.0);
            let pq1 = v1 - e1 * clamp(dot(v1, e1) / dot(e1, e1), 0.0, 1.0);
            let pq2 = v2 - e2 * clamp(dot(v2, e2) / dot(e2, e2), 0.0, 1.0);
            
            let s = sign(e0.x * e2.y - e0.y * e2.x);
            
            let d0 = vec2<f32>(dot(pq0, pq0), s * (v0.x * e0.y - v0.y * e0.x));
            let d1 = vec2<f32>(dot(pq1, pq1), s * (v1.x * e1.y - v1.y * e1.x));
            let d2 = vec2<f32>(dot(pq2, pq2), s * (v2.x * e2.y - v2.y * e2.x));
            
            let d = min(min(d0, d1), d2);
            
            return -sqrt(d.x) * sign(d.y);
        }
                
        const SHADOW_MAX_STEPS: i32 = 16;
        const SHADOW_K: f32 = 8;
        
        fn compute_shadow(ro: vec2<f32>, light_dir: vec2<f32>, instance_index: u32) -> f32 {
            var t: f32 = 0.02;
            var light: f32 = 1.0;
            for (var i: i32 = 0; i < SHADOW_MAX_STEPS; i = i + 1) {
                let pos = ro + light_dir * t;
                let dist = sd_shape(pos, instance_index);

                light = min(light, SHADOW_K * dist / t);
                t = t + dist;
            }

            return 1.0 - clamp(light, 0.0, 1.0);
        }
    `,
);

// console.log('>>', shaderMeta.shader.trim());
