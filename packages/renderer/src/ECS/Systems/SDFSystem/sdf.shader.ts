import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";

export const MAX_INSTANCE_COUNT = 10_000;

export const shaderMeta = new ShaderMeta(
  {
    // Combined view-projection (reverse-Z). Name kept as `projection`/uProjection
    // for zero churn; `uProjection * model` now reads `VP * model`.
    projection: new VariableMeta("uProjection", VariableKind.Uniform, `mat4x4<f32>`),
    // Eye position in WORLD space (ray origin for the SDF impostor). vec3 padded
    // to 16 bytes by std layout — upload from a 4-float buffer.
    cameraPos: new VariableMeta("uCameraPos", VariableKind.Uniform, `vec4<f32>`),

    transform: new VariableMeta(
      "uTransform",
      VariableKind.StorageRead,
      `array<mat4x4<f32>, ${MAX_INSTANCE_COUNT}>`,
    ),
    // Per-instance inverse model matrix (mat4.invert(uTransform[i])). Pushes the
    // world ray into canonical local SDF space; its upper-3x3 transposed is the
    // world-normal matrix (no extra upload).
    invTransform: new VariableMeta(
      "uInvTransform",
      VariableKind.StorageRead,
      `array<mat4x4<f32>, ${MAX_INSTANCE_COUNT}>`,
    ),

    // 3D ShapeKind: 10 box, 11 sphere, 12 plane.
    kind: new VariableMeta("uKind", VariableKind.StorageRead, `array<u32, ${MAX_INSTANCE_COUNT}>`),
    color: new VariableMeta(
      "uColor",
      VariableKind.StorageRead,
      `array<vec4<f32>, ${MAX_INSTANCE_COUNT}>`,
    ),
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
    intensity: new VariableMeta(
      "uIntensity",
      VariableKind.StorageRead,
      `array<f32, ${MAX_INSTANCE_COUNT}>`,
    ),
  },
  {},
  wgsl /* WGSL */ `
        // ============= ShapeKind (3D) =============
        const KIND_BOX:    u32 = 10u; // values: (hx, hy, hz)
        const KIND_SPHERE: u32 = 11u; // values: (r)
        // A flat tile is a thin Box3D — there is no separate plane primitive.

        // Rounded-box sphere-trace tuning (IQ-style). Only rounded boxes march;
        // sharp boxes + spheres are analytic. The march starts at the analytic
        // AABB entry (no wasted steps crossing empty space) and uses a RELATIVE
        // epsilon (eps grows with distance, like IQ's abs(h)<0.0001*t), so it
        // converges in few steps and does NOT starve the way a fixed-eps,
        // start-from-camera march did. Big flat surfaces never reach this path.
        // Step budget is live-tunable (debug GUI) — packed into uCameraPos.w (the
        // padding lane of the camera vec4), since it dominates zoom-in cost.
        const ROUND_REL_EPS: f32 = 1e-4;
        fn roundSteps() -> i32 { return max(i32(uCameraPos.w), 1); }

        // ============= IO =============

        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) @interpolate(flat) instance_index: u32,
            @location(1) world_pos: vec3<f32>, // interpolated cube-surface point, WORLD space
        };

        struct FragmentOutput {
            @location(0) albedo:   vec4<f32>, // rgba8unorm
            @location(1) normal:   vec4<f32>, // rgba16float, world normal in .xyz
            @location(2) emission: vec4<f32>, // rgba16float HDR
            @builtin(frag_depth) depth: f32,
        };

        // ============= uValues accessor =============
        fn val(instance_index: u32, k: u32) -> f32 {
            return uValues[instance_index * 6u + k];
        }

        // ============= Rounded-box SDF (IQ) =============
        // Only used by the rounded-box march. A rounded box of half-extent H and
        // corner radius r is sd_box(core) - r with core = H - r, so its faces
        // still reach H (the impostor cube extent) and only the corners round in.
        fn sd_box(p: vec3<f32>, b: vec3<f32>) -> f32 {
            let q = abs(p) - b;
            return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
        }
        fn sd_rbox(p: vec3<f32>, core: vec3<f32>, r: f32) -> f32 {
            return sd_box(p, core) - r;
        }
        // Surface normal via the tetrahedron gradient (4 evals).
        fn rbox_normal(p: vec3<f32>, core: vec3<f32>, r: f32) -> vec3<f32> {
            let k = vec2<f32>(1.0, -1.0);
            let e = 1e-3;
            return normalize(
                k.xyy * sd_rbox(p + k.xyy * e, core, r) +
                k.yyx * sd_rbox(p + k.yyx * e, core, r) +
                k.yxy * sd_rbox(p + k.yxy * e, core, r) +
                k.xxx * sd_rbox(p + k.xxx * e, core, r)
            );
        }

        // ============= Cube impostor =============

        // Per-instance impostor half-extents (the bounding box the cube is
        // scaled to): box half-extents, or the sphere radius on all axes.
        fn cube_half_extent(instance_index: u32) -> vec3<f32> {
            let kind = uKind[instance_index];
            if (kind == KIND_SPHERE) {
                let r = val(instance_index, 0u);
                return vec3<f32>(r, r, r);
            }
            // KIND_BOX
            return vec3<f32>(
                val(instance_index, 0u),
                val(instance_index, 1u),
                val(instance_index, 2u),
            );
        }

        // 36-vertex unit cube ([-1,1]^3) from vertex_index, no vertex buffer.
        // Faces wound CCW as seen from OUTSIDE; with frontFace:'ccw' +
        // cullMode:'front' the far faces survive, so a fragment exists even when
        // the camera is inside the cube.
        fn cube_corner(vertex_index: u32) -> vec3<f32> {
            // Verified consistent winding: every triangle is CCW as seen from
            // OUTSIDE the cube (geometric normal points away from center).
            var idx = array<u32, 36>(
                0u,2u,3u, 0u,3u,1u,   // -z
                4u,5u,7u, 4u,7u,6u,   // +z
                0u,4u,6u, 0u,6u,2u,   // -x
                1u,3u,7u, 1u,7u,5u,   // +x
                0u,1u,5u, 0u,5u,4u,   // -y
                2u,6u,7u, 2u,7u,3u    // +y
            );
            let c = idx[vertex_index];
            return vec3<f32>(
                select(-1.0, 1.0, (c & 1u) != 0u),
                select(-1.0, 1.0, (c & 2u) != 0u),
                select(-1.0, 1.0, (c & 4u) != 0u),
            );
        }

        @vertex
        fn vs_main(
            @builtin(vertex_index) vertex_index: u32,
            @builtin(instance_index) instance_index: u32,
        ) -> VertexOutput {
            let he = cube_half_extent(instance_index);
            let local = cube_corner(vertex_index) * he;
            let model = uTransform[instance_index];
            let world = model * vec4<f32>(local, 1.0);
            let clip = uProjection * world; // VP * world
            return VertexOutput(clip, instance_index, world.xyz);
        }

        // ============= Analytic ray-primitive intersection =============
        // Sphere and SHARP box are intersected in closed form (1 step, exact
        // normal, no grazing failure). Rounded boxes have no cheap closed form ->
        // they sphere-trace (IQ-style: analytic AABB entry + relative epsilon),
        // which is safe because they are small/compact. Large flat surfaces are
        // always sharp boxes -> analytic, so the old grazing bug can't return.
        struct RayHit {
            ok: bool,
            t:  f32,
            n:  vec3<f32>, // local-space surface normal
        };

        // Nearest entry into the local AABB [-he, he] + its face normal.
        fn ray_box(ro: vec3<f32>, rd: vec3<f32>, he: vec3<f32>) -> RayHit {
            let inv = 1.0 / rd;
            let t0 = (-he - ro) * inv;
            let t1 = ( he - ro) * inv;
            let tsm = min(t0, t1); // per-axis near
            let tbg = max(t0, t1); // per-axis far
            let tn = max(max(tsm.x, tsm.y), tsm.z);
            let tf = min(min(tbg.x, tbg.y), tbg.z);
            var h: RayHit;
            h.ok = tf >= max(tn, 0.0);
            h.t = max(tn, 0.0);
            // Entry face: the axis whose near-t won. Outward normal opposes rd.
            if (tn == tsm.x) { h.n = vec3<f32>(-sign(rd.x), 0.0, 0.0); }
            else if (tn == tsm.y) { h.n = vec3<f32>(0.0, -sign(rd.y), 0.0); }
            else { h.n = vec3<f32>(0.0, 0.0, -sign(rd.z)); }
            return h;
        }

        // Sphere centered at the local origin, radius r.
        fn ray_sphere(ro: vec3<f32>, rd: vec3<f32>, r: f32) -> RayHit {
            var h: RayHit;
            h.ok = false; h.t = 0.0; h.n = vec3<f32>(0.0, 1.0, 0.0);
            let b = dot(ro, rd);
            let c = dot(ro, ro) - r * r;
            let disc = b * b - c;
            if (disc < 0.0) { return h; }
            let sq = sqrt(disc);
            var t = -b - sq;
            if (t < 0.0) { t = -b + sq; } // inside: take the far root
            if (t < 0.0) { return h; }
            h.ok = true; h.t = t; h.n = normalize(ro + rd * t);
            return h;
        }

        @fragment
        fn fs_main(in: VertexOutput) -> FragmentOutput {
            let i = in.instance_index;
            let model = uTransform[i];
            let invModel = uInvTransform[i];

            // World ray: origin = camera eye, dir toward the interpolated cube
            // surface point.
            let ro_w = uCameraPos.xyz;
            let rd_w = normalize(in.world_pos - ro_w);

            // Into instance-local space. Point uses w=1; direction uses w=0 then
            // renormalize (non-uniform scale changes dir length).
            let ro = (invModel * vec4<f32>(ro_w, 1.0)).xyz;
            var rd = (invModel * vec4<f32>(rd_w, 0.0)).xyz;
            rd = rd / length(rd);

            let kind = uKind[i];
            var t = 0.0;
            var hit = false;
            var n_local = vec3<f32>(0.0, 1.0, 0.0);

            let he = vec3<f32>(val(i, 0u), val(i, 1u), val(i, 2u));
            let roundness = uRoundness[i];

            if (kind == KIND_SPHERE) {
                let h = ray_sphere(ro, rd, val(i, 0u));
                hit = h.ok; t = h.t; n_local = h.n;
            } else if (roundness <= 0.0) { // sharp box -> analytic
                let h = ray_box(ro, rd, he);
                hit = h.ok; t = h.t; n_local = h.n;
            } else { // rounded box -> IQ-style sphere-trace inside its AABB
                // Analytic entry/exit of the impostor AABB (extent = he; the
                // rounded faces still reach he, only corners round in).
                let inv = 1.0 / rd;
                let t0 = (-he - ro) * inv;
                let t1 = ( he - ro) * inv;
                let tEnter = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
                let tExit  = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
                if (tExit >= max(tEnter, 0.0)) {
                    let core = max(he - vec3<f32>(roundness), vec3<f32>(0.0));
                    t = max(tEnter, 0.0);
                    let roundStepsI = roundSteps();
                    for (var s = 0; s < roundStepsI; s = s + 1) {
                        let d = sd_rbox(ro + rd * t, core, roundness);
                        if (d < ROUND_REL_EPS * t) { hit = true; break; }
                        t = t + d;
                        if (t > tExit) { break; }
                    }
                    if (hit) { n_local = rbox_normal(ro + rd * t, core, roundness); }
                }
            }
            if (!hit) { discard; }

            let p_local = ro + rd * t;

            // Normal: local surface normal -> world via inverse-transpose of
            // model upper-3x3 == transpose of inv-model upper-3x3.
            let nrm_mat = transpose(mat3x3<f32>(invModel[0].xyz, invModel[1].xyz, invModel[2].xyz));
            let n_world = normalize(nrm_mat * n_local);

            // World hit -> reverse-Z depth.
            let world_hit = (model * vec4<f32>(p_local, 1.0)).xyz;
            let clip = uProjection * vec4<f32>(world_hit, 1.0);
            let frag_depth = clip.z / clip.w;

            let baseColor = uColor[i];
            if (baseColor.a == 0.0) { discard; }

            let intensity = uIntensity[i];          // 0 = non-emitter
            let emissive = baseColor.rgb * abs(intensity);

            return FragmentOutput(
                vec4<f32>(baseColor.rgb, baseColor.a),  // albedo
                vec4<f32>(n_world, 0.0),                // world normal (signed; rgba16float)
                vec4<f32>(emissive, 1.0),               // emission HDR
                frag_depth,
            );
        }
    `,
);

// console.log('>>', shaderMeta.shader.trim());
