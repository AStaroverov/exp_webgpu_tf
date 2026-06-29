import { wgsl } from "../../../WGSL/wgsl.ts";

// Shared local-SDF helpers used by BOTH the draw pass (sdf.shader.ts) and the
// world-space RC gather pass. Pure functions over a per-instance index; they read
// the storage buffers uKind / uValues / uRoundness by global name — any shader that
// inlines this fragment MUST declare those buffers with identical names + types
// (uValues = 8 f32 / instance). The half-height is read per-kind from uValues via
// footprint_half_z — there is no separate Height buffer.
//
// Extracted verbatim from sdf.shader.ts (the former "Helpers" + 2D/3D SDF blocks).
// The wgsl`` tag inlines a no-`name` fragment's `.body` in-place, so the emitted
// shader string is byte-identical to the inline version.
export const sceneSDF = wgsl /* WGSL */ `
        // ============= Helpers =============

        // Per-instance rotation basis from the rigid transform's columns. normalize()
        // defensively drops any accidental uniform scale; for the rigid case (the only
        // case here) the columns are already unit. transpose(R) = inverse(R) for an
        // orthonormal R: world->local dir = transpose(R)*v, local->world = R*v.
        fn instance_rot(t: mat4x4<f32>) -> mat3x3<f32> {
            return mat3x3<f32>(normalize(t[0].xyz), normalize(t[1].xyz), normalize(t[2].xyz));
        }

        // Uniform scale baked into the rigid transform: every column has length s, so
        // column 0 alone gives it. The SDF is evaluated at the UNSCALED footprint, so the
        // draw path divides the local point by s and multiplies the hit point back by s.
        fn instance_scale(t: mat4x4<f32>) -> f32 {
            return length(t[0].xyz);
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
            let width = uValues[instance_index * 8u + 0u];
            let height = uValues[instance_index * 8u + 1u];
            let roundness = uRoundness[instance_index];
            var dist = 1.0;

            if (kind == 0u) {
                // Circle -> cylinder footprint.
                dist = sd_circle(p, width / 2.0);
            } else if (kind == 1u) {
                dist = sd_rectangle(p, width / 2.0 - roundness, height / 2.0 - roundness);
            } else if (kind == 3u) {
                dist = sd_parallelogram(p, width / 2.0 - roundness, height / 2.0 - roundness, uValues[instance_index * 8u + 2u]);
            } else if (kind == 4u) {
                // Trapezoid: values = [topWidth, bottomWidth, height].
                // sd_trapezoid(p, r1=bottom half-width, r2=top half-width, he=half-height).
                dist = sd_trapezoid(
                    p,
                    uValues[instance_index * 8u + 1u] / 2.0 - roundness,
                    uValues[instance_index * 8u + 0u] / 2.0 - roundness,
                    uValues[instance_index * 8u + 2u] / 2.0 - roundness,
                );
            } else if (kind == 5u) {
                let ax = uValues[instance_index * 8u + 0u] - sign(uValues[instance_index * 8u + 0u]) * roundness;
                let ay = uValues[instance_index * 8u + 1u] - sign(uValues[instance_index * 8u + 1u]) * roundness;
                let bx = uValues[instance_index * 8u + 2u] - sign(uValues[instance_index * 8u + 2u]) * roundness;
                let by = uValues[instance_index * 8u + 3u] - sign(uValues[instance_index * 8u + 3u]) * roundness;
                let cx = uValues[instance_index * 8u + 4u] - sign(uValues[instance_index * 8u + 4u]) * roundness;
                let cy = uValues[instance_index * 8u + 5u] - sign(uValues[instance_index * 8u + 5u]) * roundness;
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

        // Half of the shape's Z extent, read per-kind from uValues' depth slot. Sphere
        // (6) has no extrusion: its radius is already its full half-extent.
        fn footprint_half_z(instance_index: u32) -> f32 {
            let kind = uKind[instance_index];
            if (kind == 0u) {
                return uValues[instance_index * 8u + 1u] * 0.5;
            } else if (kind == 1u) {
                return uValues[instance_index * 8u + 2u] * 0.5;
            } else if (kind == 3u) {
                return uValues[instance_index * 8u + 3u] * 0.5;
            } else if (kind == 4u) {
                return uValues[instance_index * 8u + 3u] * 0.5;
            } else if (kind == 5u) {
                return uValues[instance_index * 8u + 6u] * 0.5;
            }
            return uValues[instance_index * 8u + 0u];
        }

        fn sd_shape3d(p: vec3<f32>, instance_index: u32) -> f32 {
            if (uKind[instance_index] == 6u) {
                // True 3D sphere; values[0] = radius.
                return length(p) - uValues[instance_index * 8u + 0u];
            }
            let d2 = sd_2d_for_kind(p.xy, instance_index);
            return extrude(d2, p.z, footprint_half_z(instance_index));
        }

        fn sd_normal3d(p: vec3<f32>, instance_index: u32) -> vec3<f32> {
            let e = vec2<f32>(0.0015, -0.0015);
            return normalize(
                e.xyy * sd_shape3d(p + e.xyy, instance_index) +
                e.yyx * sd_shape3d(p + e.yyx, instance_index) +
                e.yxy * sd_shape3d(p + e.yxy, instance_index) +
                e.xxx * sd_shape3d(p + e.xxx, instance_index)
            );
        }

        // Per-kind XY footprint half-extents (ported from the 2D compute_rect_vertex
        // bounding-box logic), so the impostor box silhouette never clips the SDF.
        fn footprint_half_xy(instance_index: u32) -> vec2<f32> {
            let kind = uKind[instance_index];
            var width = uValues[instance_index * 8u + 0u];
            var height = uValues[instance_index * 8u + 1u];

            if (kind == 0u) {
                // Circle / cylinder: square footprint from the radius (values[0] = radius).
                height = width;
            } else if (kind == 6u) {
                // Sphere: values[0] = radius → full extent 2r in both axes.
                width = uValues[instance_index * 8u + 0u] * 2.0;
                height = width;
            } else if (kind == 3u) {
                // Parallelogram: footprint widens by the skew amount on each side.
                width += abs(uValues[instance_index * 8u + 2u]) * 2.0;
            } else if (kind == 4u) {
                // Trapezoid: values = [topWidth, bottomWidth, height]. Footprint bound =
                // wider of the two ends in X, the height value in Y.
                width = max(uValues[instance_index * 8u + 0u], uValues[instance_index * 8u + 1u]);
                height = uValues[instance_index * 8u + 2u];
            } else if (kind == 5u) {
                // Triangle: bounding box from the three vertices.
                width = max(width, max(uValues[instance_index * 8u + 2u], uValues[instance_index * 8u + 4u])) * 2.0;
                height = max(height, max(uValues[instance_index * 8u + 3u], uValues[instance_index * 8u + 5u])) * 2.0;
            }

            return vec2<f32>(width / 2.0, height / 2.0);
        }
`;
