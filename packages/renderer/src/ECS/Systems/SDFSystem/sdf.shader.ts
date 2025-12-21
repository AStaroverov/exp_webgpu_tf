import { VariableKind, VariableMeta } from '../../../Struct/VariableMeta.ts';
import { ShaderMeta } from '../../../WGSL/ShaderMeta.ts';
import { wgsl } from '../../../WGSL/wgsl.ts';

export const MAX_INSTANCE_COUNT = 10_000;

export const shaderMeta = new ShaderMeta(
    {
        projection: new VariableMeta('uProjection', VariableKind.Uniform, `mat4x4<f32>`),
        transform: new VariableMeta('uTransform', VariableKind.StorageRead, `array<mat4x4<f32>, ${ MAX_INSTANCE_COUNT }>`),

        // 0: circle, 1: rectangle, 2: rhombus
        kind: new VariableMeta('uKind', VariableKind.StorageRead, `array<u32, ${ MAX_INSTANCE_COUNT }>`),
        color: new VariableMeta('uColor', VariableKind.StorageRead, `array<vec4<f32>, ${ MAX_INSTANCE_COUNT }>`),
        values: new VariableMeta('uValues', VariableKind.StorageRead, `array<f32, ${ MAX_INSTANCE_COUNT * 6 }>`),
        roundness: new VariableMeta('uRoundness', VariableKind.StorageRead, `array<f32, ${ MAX_INSTANCE_COUNT }>`),

        // Note: r32float is unfilterable, so we use textureLoad instead of textureSample
        // Put in separate group (2) so it can be excluded from shadow map pass
        shadowMap: new VariableMeta('uShadowMap', VariableKind.Texture, 'texture_2d<f32>', {
            group: 2,
            visibility: GPUShaderStage.FRAGMENT,
            textureSampleType: 'unfilterable-float',
        }),
    },
    {},
    wgsl/* WGSL */`
        // ============= Structures =============
        
        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) @interpolate(flat) instance_index: u32,
            @location(1) local_position: vec2<f32>,
        };
        
        struct FragmentOutput {
            @location(0) color : vec4<f32>,
            @builtin(frag_depth) depth : f32,
        };
        
        // ============= Shadow Constants =============
        
        // Light direction constant (normalized, pointing from light source)
        const LIGHT_DIR: vec2<f32> = vec2<f32>(-0.5, -0.5);
        // Shadow darkness (0 = invisible, 1 = fully black)
        const SHADOW_DARKNESS: f32 = 0.4;
        // Minimum Z difference to apply shadow
        const SHADOW_Z_THRESHOLD: f32 = 0.01;
        // Visual shadow glow size relative to Z (0.5 = half of shadow offset)
        const SHADOW_GLOW_RATIO: f32 = 1;
        // Ray marching constants for soft shadows
        const SHADOW_MAX_STEPS: i32 = 16;
        const SHADOW_SOFTNESS: f32 = 8.0;
        
        // ============= Common Shadow Helpers =============
        
        // Computes shadow offset in world space based on object height
        fn compute_shadow_offset(z_height: f32) -> vec2<f32> {
            return -LIGHT_DIR * z_height;
        }
        
        // Projects world position to clip space with Y-flip
        fn project_to_clip(world_pos: vec2<f32>) -> vec4<f32> {
            let projected = (uProjection * vec4<f32>(world_pos, 0.0, 1.0)).xy;
            return vec4<f32>(projected.x, -projected.y, 0.0, 1.0);
        }
        
        // Transforms local vertex to world position with shadow offset
        fn transform_shadow_vertex(rect_vertex: vec2<f32>, instance_index: u32, z_height: f32) -> vec4<f32> {
            let transform = uTransform[instance_index];
            let world_pos = (transform * vec4<f32>(rect_vertex, 0.0, 1.0)).xy + compute_shadow_offset(z_height);
            return project_to_clip(world_pos);
        }
        
        // ============= Main Shape Pass =============
        // Renders shapes with shadow map sampling for object-to-object shadows
        
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
            @builtin(position) frag_coord: vec4<f32>,
            @location(0) @interpolate(flat) instance_index: u32,
            @location(1) local_position: vec2<f32>,
        ) -> FragmentOutput {
            let dist = sd_shape(local_position, instance_index);
            let color = uColor[instance_index];
            
            if (dist > 0.0 || color.a == 0.0) { 
                discard;
            }
            
            let object_z = uTransform[instance_index][3].z;
            
            // Load shadow map at current screen position (r32float is unfilterable, use textureLoad)
            let shadow_z = textureLoad(uShadowMap, vec2<i32>(frag_coord.xy), 0).r;
            
            // Apply shadow if shadow caster is above this object
            var final_color = color;
            if (shadow_z > object_z + SHADOW_Z_THRESHOLD) {
                final_color = vec4<f32>(color.rgb * (1.0 - SHADOW_DARKNESS), color.a);
            }

            return FragmentOutput(final_color, object_z);
        }

        // ============= Shadow Map Pass =============
        // Renders shadow silhouettes to texture, outputs Z height of shadow caster

        @vertex
        fn vs_shadow_map(
            @builtin(vertex_index) vertex_index: u32,
            @builtin(instance_index) instance_index: u32
        ) -> VertexOutput {
            let z_height = uTransform[instance_index][3].z;
            
            // Skip objects at ground level
            if (z_height <= SHADOW_Z_THRESHOLD) {
                return VertexOutput(vec4<f32>(0.0), instance_index, vec2<f32>(0.0));
            }
            
            let rect_vertex = compute_rect_vertex(vertex_index, instance_index);
            let position = transform_shadow_vertex(rect_vertex, instance_index, z_height);
            
            return VertexOutput(position, instance_index, rect_vertex);
        }
        
        @fragment
        fn fs_shadow_map(
            @location(0) @interpolate(flat) instance_index: u32,
            @location(1) local_position: vec2<f32>,
        ) -> @location(0) f32 {
            let z_height = uTransform[instance_index][3].z;
            
            if (z_height <= SHADOW_Z_THRESHOLD) {
                discard;
            }
            
            let dist = sd_shape(local_position, instance_index);
        
            if (dist > 0.0) {
                discard;
            }
            
            return z_height;
        }

        // ============= Visual Shadow Pass =============
        // Renders soft shadow glow effect on the ground plane
        
        @vertex
        fn vs_shadow(
            @builtin(vertex_index) vertex_index: u32,
            @builtin(instance_index) instance_index: u32
        ) -> VertexOutput {
            let z_height = uTransform[instance_index][3].z;
            
            // Compute glow size from Z - higher objects have larger glow
            let glow_size = z_height * SHADOW_GLOW_RATIO;
            
            let original_vertex = compute_rect_vertex(vertex_index, instance_index);
            let rect_vertex = original_vertex + normalize(original_vertex) * glow_size;
            let position = transform_shadow_vertex(rect_vertex, instance_index, z_height);
            
            return VertexOutput(position, instance_index, rect_vertex);
        }    

        @fragment
        fn fs_shadow(
            @location(0) @interpolate(flat) instance_index: u32,
            @location(1) local_position: vec2<f32>,
        ) -> FragmentOutput {
            let z_height = uTransform[instance_index][3].z;
           
            if (z_height <= SHADOW_Z_THRESHOLD) {
                discard;
            }
            
            let glow_size = z_height * SHADOW_GLOW_RATIO;
            let dist = sd_shape(local_position, instance_index);

            // Compute light direction in local space
            let transform = uTransform[instance_index];
            let rotation = mat2x2<f32>(transform[0].xy, transform[1].xy);
            let local_light_dir = normalize(LIGHT_DIR * rotation);

            let shadow_intensity = compute_shadow_intensity(local_position, local_light_dir, instance_index);
            let edge_fade = select(1.0, 1.0 - smoothstep(0.0, glow_size, abs(dist)), dist > 0.0);
        
            // Shadow is rendered at ground level (depth = 0)
            return FragmentOutput(
                vec4<f32>(uColor[instance_index].rgb * 0.2, shadow_intensity * edge_fade),
                0.0
            );
        }
        
        // ============= Geometry Helpers =============
        
        fn compute_rect_vertex(vertex_index: u32, instance_index: u32) -> vec2<f32> {
            let kind = uKind[instance_index];
            var width = uValues[instance_index*6+0];
            var height = uValues[instance_index*6+1];
            
            // Adjust bounds based on shape type
            if (kind == 0u) {
                // Circle: use width for both dimensions
                height = width;
            } else if (kind == 3u) {
                // Parallelogram: extend width by skew amount
                width += abs(uValues[instance_index*6+2]) * 2.0;
            } else if (kind == 5u) {
                // Triangle: compute bounding box from vertices
                width = max(width, max(uValues[instance_index*6+2], uValues[instance_index*6+4])) * 2.0;
                height = max(height, max(uValues[instance_index*6+3], uValues[instance_index*6+5])) * 2.0;
            }

            let half_size = vec2<f32>(width / 2.0, height / 2.0);
            
            return vec2<f32>(
                select(-half_size.x, half_size.x, vertex_index > 0u && vertex_index < 4u),
                select(-half_size.y, half_size.y, vertex_index > 1u && vertex_index < 5u),
            );
        }

        fn to_final_position(transform: mat4x4<f32>, pos: vec2<f32>) -> vec2<f32> {
            let res = (uProjection * transform * vec4<f32>(pos, 0.0, 1.0)).xy; 
            return vec2<f32>(res.x, -res.y);
        }
        
        // ============= SDF Shape Functions =============
            
        fn op_round(d: f32, r: f32) -> f32 {
            return d - r;
        }
        
        fn sd_shape(pos: vec2<f32>, instance_index: u32) -> f32 {
            let kind = uKind[instance_index];
            let width = uValues[instance_index*6+0];
            let height = uValues[instance_index*6+1];
            let roundness = uRoundness[instance_index];
            var dist = 1.0;
 
            if (kind == 0u) {
                dist = sd_circle(pos, width / 2.0);
            } else if (kind == 1u) {
                dist = sd_rectangle(pos, width / 2.0 - roundness, height / 2.0 - roundness);
            } else if (kind == 2u) {
                dist = sd_rhombus(pos, width / 2.0 - roundness, height / 2.0 - roundness);
            } else if (kind == 3u) {
                dist = sd_parallelogram(pos, width / 2.0 - roundness, height / 2.0 - roundness, uValues[instance_index*6+2]);
            } else if (kind == 4u) {
                dist = sd_trapezoid(pos, width / 2.0 - roundness, uValues[instance_index*6+2] / 2.0 - roundness, height / 2.0 - roundness);
            } else if (kind == 5u) {
                let ax = uValues[instance_index*6+0] - sign(uValues[instance_index*6+0]) * roundness;
                let ay = uValues[instance_index*6+1] - sign(uValues[instance_index*6+1]) * roundness;
                let bx = uValues[instance_index*6+2] - sign(uValues[instance_index*6+2]) * roundness;
                let by = uValues[instance_index*6+3] - sign(uValues[instance_index*6+3]) * roundness;
                let cx = uValues[instance_index*6+4] - sign(uValues[instance_index*6+4]) * roundness;
                let cy = uValues[instance_index*6+5] - sign(uValues[instance_index*6+5]) * roundness;
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
        
        fn sd_rectangle(p: vec2<f32>, w: f32, h: f32) -> f32 {
            let b = vec2(w, h);
            let d = abs(p) - b;
            return length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0);
        }
        
        fn ndot(a: vec2<f32>, b: vec2<f32>) -> f32 {
            return a.x * b.x - a.y * b.y;
        } 
        
        fn sd_rhombus(_p: vec2<f32>, wi: f32, he: f32) -> f32 {
            let p = abs(_p);
            let b = vec2<f32>(wi, he);
            let h = clamp(ndot(b - 2.0 * p, b) / dot(b, b), -1.0, 1.0);
            let d = length(p - 0.5 * b * vec2(1.0 - h, 1.0 + h));
            return d * sign(p.x * b.y + p.y * b.x - b.x * b.y);
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
            let ca = vec2<f32>(max(0.0, pp.x - selected_r), abs(pp.y) - he);
            let cb = pp - k1 + k2 * clamp(dot(k1 - pp, k2) / dot2(k2), 0.0, 1.0);
            let s = select(1.0, -1.0, cb.x < 0.0 && ca.y < 0.0);
            return s * sqrt(min(dot2(ca), dot2(cb)));
        }
        
        fn sd_parallelogram(p: vec2<f32>, wi: f32, he: f32, sk: f32) -> f32 {
            let e = vec2<f32>(sk, he);
            let e2 = sk * sk + he * he;
            var pos = select(p, -p, p.y < 0.0);
            // Horizontal edge
            var w = pos - e;
            w.x = w.x - clamp(w.x, -wi, wi);
            var d = vec2<f32>(dot(w, w), -w.y);
            // Vertical edge
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
        
        // ============= Soft Shadow Ray Marching =============
        
        fn compute_shadow_intensity(origin: vec2<f32>, light_dir: vec2<f32>, instance_index: u32) -> f32 {
            var t = 0.02;
            var light = 1.0;
            
            for (var i = 0; i < SHADOW_MAX_STEPS; i++) {
                let pos = origin + light_dir * t;
                let dist = sd_shape(pos, instance_index);
                light = min(light, SHADOW_SOFTNESS * dist / t);
                t += dist;
            }

            return 1.0 - clamp(light, 0.0, 1.0);
        }
    `,
);

// console.log('>>', shaderMeta.shader.trim());
