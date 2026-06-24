import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";

// 2.5D true-3D-SDF draw pass (ported from prototype/sdf3d/shader.ts).
//
// One instanced draw call. Per instance we rasterize a world-space bounding box
// (the "impostor": per-kind XY footprint half-extents + half-height hz), and in
// the fragment shader sphere-trace the instance's local 3D SDF along the (constant,
// orthographic) camera ray. The ray hit gives a real world point → we write its
// clip depth to frag_depth so the depth buffer sorts/occludes correctly with zero
// CPU sorting. One directional light shades the surface normal.
//
// SHAPE KINDS (src/ECS/Components/Shape.ts ShapeKind): Circle=0 (extrudes to a
// CYLINDER), Rectangle=1, Parallelogram=3, Trapezoid=4, Triangle=5, Sphere=6
// (the ONLY non-extruded primitive: length(p) - r). Everything else is the exact
// 2D footprint SDF extruded by the entity's Height.
//
// HEIGHT MODEL: transform col3.z = baseZ (bottom). center.z = baseZ + height*0.5,
// half-height hz = height*0.5.
//
// DEPTH CONVENTION — REVERSE-Z. The pipeline uses depthCompare "greater-equal"
// with depthClearValue 0 (GPUShader.ts withDepth + createFrame.ts), so nearer =
// LARGER depth. viewProj (built in ResizeSystem.ts with an ndc-fix) already maps
// NEAR→1, FAR→0; we just write frag_depth = (viewProj * hitWorld).z / .w and the
// reverse-Z range falls out. This MUST match ResizeSystem.viewProjMatrix.

export const MAX_INSTANCE_COUNT = 10_000;

export const shaderMeta = new ShaderMeta(
  {
    // ---- group 0 : uniforms ----
    viewProj: new VariableMeta("uViewProj", VariableKind.Uniform, `mat4x4<f32>`),
    // xyz = world-space camera forward (normalized); w unused.
    rayDir: new VariableMeta("uRayDir", VariableKind.Uniform, `vec4<f32>`),
    // xyz = world-space light direction (points along travel); w unused.
    lightDir: new VariableMeta("uLightDir", VariableKind.Uniform, `vec4<f32>`),

    // ---- group 1 : per-instance storage ----
    // Full per-instance transform: center = col3.xyz, yaw from upper-left 2x2.
    transform: new VariableMeta(
      "uTransform",
      VariableKind.StorageRead,
      `array<mat4x4<f32>, ${MAX_INSTANCE_COUNT}>`,
    ),
    kind: new VariableMeta("uKind", VariableKind.StorageRead, `array<u32, ${MAX_INSTANCE_COUNT}>`),
    values: new VariableMeta(
      "uValues",
      VariableKind.StorageRead,
      `array<f32, ${MAX_INSTANCE_COUNT * 6}>`,
    ),
    roundness: new VariableMeta(
      "uRoundness",
      VariableKind.StorageRead,
      `array<f32, ${MAX_INSTANCE_COUNT}>`,
    ),
    // Honest vertical extent per instance (world units).
    heights: new VariableMeta(
      "uHeights",
      VariableKind.StorageRead,
      `array<f32, ${MAX_INSTANCE_COUNT}>`,
    ),
    color: new VariableMeta(
      "uColor",
      VariableKind.StorageRead,
      `array<vec4<f32>, ${MAX_INSTANCE_COUNT}>`,
    ),
  },
  {},
  wgsl /* WGSL */ `
        // ============= Structures =============

        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) @interpolate(flat) instance_index: u32,
            @location(1) world: vec3<f32>,
        };

        struct FragmentOutput {
            @location(0) color : vec4<f32>,
            @builtin(frag_depth) depth : f32,
        };

        // ============= Helpers =============

        fn rotZ(p: vec2<f32>, c: f32, s: f32) -> vec2<f32> {
            return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
        }

        fn op_round(d: f32, r: f32) -> f32 {
            return d - r;
        }

        fn ndot(a: vec2<f32>, b: vec2<f32>) -> f32 {
            return a.x * b.x - a.y * b.y;
        }

        fn dot2(v: vec2<f32>) -> f32 {
            return dot(v, v);
        }

        // ============= 2D SDF footprints (verbatim from the 2D shader) =============

        fn sd_circle(p: vec2<f32>, r: f32) -> f32 {
            return length(p) - r;
        }

        fn sd_rectangle(p: vec2<f32>, w: f32, h: f32) -> f32 {
            let b = vec2(w, h);
            let d = abs(p) - b;
            return length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0);
        }

        fn sd_rhombus(_p: vec2<f32>, wi: f32, he: f32) -> f32 {
            let p = abs(_p);
            let b = vec2<f32>(wi, he);
            let h = clamp(ndot(b - 2.0 * p, b) / dot(b, b), -1.0, 1.0);
            let d = length(p - 0.5 * b * vec2(1.0 - h, 1.0 + h));
            return d * sign(p.x * b.y + p.y * b.x - b.x * b.y);
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

        // Dispatch the 2D footprint SDF by ShapeKind, reusing the EXACT 2D values
        // layout + roundness convention from the original 2D shader. (Sphere=6 is
        // handled separately in sd_shape3d — never reaches here.)
        fn sd_2d_for_kind(p: vec2<f32>, instance_index: u32) -> f32 {
            let kind = uKind[instance_index];
            let width = uValues[instance_index * 6u + 0u];
            let height = uValues[instance_index * 6u + 1u];
            let roundness = uRoundness[instance_index];
            var dist = 1.0;

            if (kind == 0u) {
                // Circle -> cylinder footprint.
                dist = sd_circle(p, width / 2.0);
            } else if (kind == 1u) {
                dist = sd_rectangle(p, width / 2.0 - roundness, height / 2.0 - roundness);
            } else if (kind == 3u) {
                dist = sd_parallelogram(p, width / 2.0 - roundness, height / 2.0 - roundness, uValues[instance_index * 6u + 2u]);
            } else if (kind == 4u) {
                // Trapezoid: values = [topWidth, bottomWidth, height].
                // sd_trapezoid(p, r1=bottom half-width, r2=top half-width, he=half-height).
                dist = sd_trapezoid(
                    p,
                    uValues[instance_index * 6u + 1u] / 2.0 - roundness,
                    uValues[instance_index * 6u + 0u] / 2.0 - roundness,
                    uValues[instance_index * 6u + 2u] / 2.0 - roundness,
                );
            } else if (kind == 5u) {
                let ax = uValues[instance_index * 6u + 0u] - sign(uValues[instance_index * 6u + 0u]) * roundness;
                let ay = uValues[instance_index * 6u + 1u] - sign(uValues[instance_index * 6u + 1u]) * roundness;
                let bx = uValues[instance_index * 6u + 2u] - sign(uValues[instance_index * 6u + 2u]) * roundness;
                let by = uValues[instance_index * 6u + 3u] - sign(uValues[instance_index * 6u + 3u]) * roundness;
                let cx = uValues[instance_index * 6u + 4u] - sign(uValues[instance_index * 6u + 4u]) * roundness;
                let cy = uValues[instance_index * 6u + 5u] - sign(uValues[instance_index * 6u + 5u]) * roundness;
                dist = sd_triangle(p, vec2f(ax, ay), vec2f(bx, by), vec2f(cx, cy));
            }

            // Circle has no roundness offset (matches the original 2D path).
            if (kind != 0u) {
                dist = op_round(dist, roundness);
            }

            return dist;
        }

        // ============= 3D SDF: sphere (kind 6) or extruded footprint =============

        fn extrude(d2: f32, pz: f32, half_h: f32) -> f32 {
            let w = vec2<f32>(d2, abs(pz) - half_h);
            return min(max(w.x, w.y), 0.0) + length(max(w, vec2<f32>(0.0)));
        }

        fn sd_shape3d(p: vec3<f32>, instance_index: u32, hz: f32) -> f32 {
            if (uKind[instance_index] == 6u) {
                // True 3D sphere; values[0] = radius.
                return length(p) - uValues[instance_index * 6u + 0u];
            }
            let d2 = sd_2d_for_kind(p.xy, instance_index);
            return extrude(d2, p.z, hz);
        }

        fn sd_normal3d(p: vec3<f32>, instance_index: u32, hz: f32) -> vec3<f32> {
            let e = vec2<f32>(0.0015, -0.0015);
            return normalize(
                e.xyy * sd_shape3d(p + e.xyy, instance_index, hz) +
                e.yyx * sd_shape3d(p + e.yyx, instance_index, hz) +
                e.yxy * sd_shape3d(p + e.yxy, instance_index, hz) +
                e.xxx * sd_shape3d(p + e.xxx, instance_index, hz)
            );
        }

        // Per-kind XY footprint half-extents (ported from the 2D compute_rect_vertex
        // bounding-box logic), so the impostor box silhouette never clips the SDF.
        fn footprint_half_xy(instance_index: u32) -> vec2<f32> {
            let kind = uKind[instance_index];
            var width = uValues[instance_index * 6u + 0u];
            var height = uValues[instance_index * 6u + 1u];

            if (kind == 0u) {
                // Circle / cylinder: square footprint from the radius (values[0] = radius).
                height = width;
            } else if (kind == 6u) {
                // Sphere: values[0] = radius → full extent 2r in both axes.
                width = uValues[instance_index * 6u + 0u] * 2.0;
                height = width;
            } else if (kind == 3u) {
                // Parallelogram: footprint widens by the skew amount on each side.
                width += abs(uValues[instance_index * 6u + 2u]) * 2.0;
            } else if (kind == 4u) {
                // Trapezoid: values = [topWidth, bottomWidth, height]. Footprint bound =
                // wider of the two ends in X, the height value in Y.
                width = max(uValues[instance_index * 6u + 0u], uValues[instance_index * 6u + 1u]);
                height = uValues[instance_index * 6u + 2u];
            } else if (kind == 5u) {
                // Triangle: bounding box from the three vertices.
                width = max(width, max(uValues[instance_index * 6u + 2u], uValues[instance_index * 6u + 4u])) * 2.0;
                height = max(height, max(uValues[instance_index * 6u + 3u], uValues[instance_index * 6u + 5u])) * 2.0;
            }

            return vec2<f32>(width / 2.0, height / 2.0);
        }

        // ============= Vertex: rasterize the per-instance impostor box =============

        // 36-vertex unit cube ([-1,1]^3), 12 triangles. Indexed by vertex_index.
        const CUBE = array<vec3<f32>, 36>(
            // -z
            vec3<f32>(-1,-1,-1), vec3<f32>( 1,-1,-1), vec3<f32>( 1, 1,-1),
            vec3<f32>(-1,-1,-1), vec3<f32>( 1, 1,-1), vec3<f32>(-1, 1,-1),
            // +z
            vec3<f32>(-1,-1, 1), vec3<f32>( 1, 1, 1), vec3<f32>( 1,-1, 1),
            vec3<f32>(-1,-1, 1), vec3<f32>(-1, 1, 1), vec3<f32>( 1, 1, 1),
            // -y
            vec3<f32>(-1,-1,-1), vec3<f32>(-1,-1, 1), vec3<f32>( 1,-1, 1),
            vec3<f32>(-1,-1,-1), vec3<f32>( 1,-1, 1), vec3<f32>( 1,-1,-1),
            // +y
            vec3<f32>(-1, 1,-1), vec3<f32>( 1, 1,-1), vec3<f32>( 1, 1, 1),
            vec3<f32>(-1, 1,-1), vec3<f32>( 1, 1, 1), vec3<f32>(-1, 1, 1),
            // -x
            vec3<f32>(-1,-1,-1), vec3<f32>(-1, 1,-1), vec3<f32>(-1, 1, 1),
            vec3<f32>(-1,-1,-1), vec3<f32>(-1, 1, 1), vec3<f32>(-1,-1, 1),
            // +x
            vec3<f32>( 1,-1,-1), vec3<f32>( 1,-1, 1), vec3<f32>( 1, 1, 1),
            vec3<f32>( 1,-1,-1), vec3<f32>( 1, 1, 1), vec3<f32>( 1, 1,-1),
        );

        @vertex
        fn vs_main(
            @builtin(vertex_index) vertex_index: u32,
            @builtin(instance_index) instance_index: u32
        ) -> VertexOutput {
            let transform = uTransform[instance_index];
            let baseZ = transform[3].z;
            let height = uHeights[instance_index];
            let hz = height * 0.5;
            // Object center sits half a height above the base.
            let center = vec3<f32>(transform[3].x, transform[3].y, baseZ + hz);

            // Yaw from the upper-left 2x2 of the transform.
            let yaw = atan2(transform[0].y, transform[0].x);

            let halfXY = footprint_half_xy(instance_index);
            let corner = CUBE[vertex_index];

            // Unit-cube corner scaled by footprint half-extents + half-height,
            // rotated by yaw into world.
            let scaled = corner * vec3<f32>(halfXY.x, halfXY.y, hz);
            let xy = rotZ(scaled.xy, cos(yaw), sin(yaw));
            let world = center + vec3<f32>(xy, scaled.z);

            var out: VertexOutput;
            out.position = uViewProj * vec4<f32>(world, 1.0);
            out.instance_index = instance_index;
            out.world = world;
            return out;
        }

        // ============= Fragment: sphere-trace the local 3D SDF =============

        @fragment
        fn fs_main(
            @builtin(position) frag_coord: vec4<f32>,
            @location(0) @interpolate(flat) instance_index: u32,
            @location(1) world: vec3<f32>,
        ) -> FragmentOutput {
            let color = uColor[instance_index];
            if (color.a == 0.0) {
                discard;
            }

            let transform = uTransform[instance_index];
            let height = uHeights[instance_index];
            let hz = height * 0.5;
            let center = vec3<f32>(transform[3].x, transform[3].y, transform[3].z + hz);
            let yaw = atan2(transform[0].y, transform[0].x);

            // World ray (origin = this fragment's box-surface point) → instance-local
            // space (remove yaw; rigid transform preserves distances).
            let ic = cos(-yaw);
            let is = sin(-yaw);
            let relW = world - center;
            let lo = vec3<f32>(rotZ(relW.xy, ic, is), relW.z);
            let ld = normalize(vec3<f32>(rotZ(uRayDir.xy, ic, is), uRayDir.z));

            // Slab test against the local AABB (footprint half-extents + half-height).
            let halfXY = footprint_half_xy(instance_index);
            let half = vec3<f32>(halfXY.x, halfXY.y, hz);
            // Guard the reciprocal: lo lies on an AABB face, so for a near-axis-aligned
            // ray (ld component ~0 as the camera orbits) (half - lo) * (1/ld) would be
            // 0 * Inf = NaN and corrupt the slab reduction.
            let safeLd = select(ld, vec3<f32>(1e-6), abs(ld) < vec3<f32>(1e-6));
            let inv = 1.0 / safeLd;
            let tA = (-half - lo) * inv;
            let tB = (half - lo) * inv;
            let tmin = min(tA, tB);
            let tmax = max(tA, tB);
            let t0 = max(max(tmin.x, tmin.y), tmin.z);
            let t1 = min(min(tmax.x, tmax.y), tmax.z);
            if (t1 < max(t0, 0.0)) {
                discard;
            }

            // Sphere-trace between entry and exit.
            var t = max(t0, 0.0);
            var hit = false;
            for (var i = 0; i < 96; i = i + 1) {
                let d = sd_shape3d(lo + ld * t, instance_index, hz);
                if (d < 0.001) {
                    hit = true;
                    break;
                }
                t = t + d;
                if (t > t1) {
                    break;
                }
            }
            if (!hit) {
                discard;
            }

            let pLocal = lo + ld * t;
            let nLocal = sd_normal3d(pLocal, instance_index, hz);

            // Back to world (forward yaw).
            let fc = cos(yaw);
            let fs_ = sin(yaw);
            let nWorld = normalize(vec3<f32>(rotZ(nLocal.xy, fc, fs_), nLocal.z));
            let pWorld = center + vec3<f32>(rotZ(pLocal.xy, fc, fs_), pLocal.z);

            let clip = uViewProj * vec4<f32>(pWorld, 1.0);

            // Ambient + clamped diffuse against one directional light.
            let diff = max(dot(nWorld, -uLightDir.xyz), 0.0);
            let ambient = 0.28;
            let shade = ambient + (1.0 - ambient) * diff;

            var out: FragmentOutput;
            out.color = vec4<f32>(color.rgb * shade, 1.0);
            // Reverse-Z: viewProj already maps NEAR→1, FAR→0 to match the pipeline's
            // greater-equal/clear-0 depth compare. See ResizeSystem.viewProjMatrix.
            out.depth = clip.z / clip.w;
            return out;
        }
    `,
);

// console.log('>>', shaderMeta.shader.trim());
