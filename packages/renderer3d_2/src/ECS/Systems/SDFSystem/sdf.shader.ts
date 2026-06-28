import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";
import { sceneSDF } from "./sceneSDF.wgsl.ts";

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
// 2D footprint SDF extruded by its per-kind depth slot in Shape.values.
//
// HEIGHT MODEL: transform col3.z = the body CENTER. The 3D SDF is centered at
// local z=0 (extrude is symmetric about z=0; sphere is length(p)-r). Half-height
// hz = footprint_half_z(instance) reads the depth slot from uValues.
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
      `array<f32, ${MAX_INSTANCE_COUNT * 8}>`,
    ),
    roundness: new VariableMeta(
      "uRoundness",
      VariableKind.StorageRead,
      `array<f32, ${MAX_INSTANCE_COUNT}>`,
    ),
    color: new VariableMeta(
      "uColor",
      VariableKind.StorageRead,
      `array<vec4<f32>, ${MAX_INSTANCE_COUNT}>`,
    ),

    // ---- group 1 : per-instance storage, emission pass only ----
    // Emission material, packed into one vec4 to stay under the 8-storage-buffer
    // per-stage limit (fs_emit already uses 7 buffers): .x = intensity (0 = occluder,
    // !=0 = emitter, <0 = directional), .y = translucency (0 opaque .. 1 transparent),
    // .z = blurness edge-feather width (local SDF units; 0 = hard edge), .w = unused.
    material: new VariableMeta(
      "uMaterial",
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

        // Stage-3b G-buffer. The main pass no longer shades: lighting comes from
        // the Radiance Cascades composite. We output raw albedo + the world normal
        // so the composite can compute a normal-aware directional term.
        //   @location(0) albedo   : raw uColor.rgb (a = 1 surface present)
        //   @location(1) normal   : vec4(nWorld*0.5+0.5, 1.0) — a = 1 surface mask
        //                           (background stays cleared 0 → mask 0).
        //   @location(2) emission : per-pixel self-emission = uColor.rgb *
        //                           abs(uMaterial.x) (a = 1; pure occluder material.x==0
        //                           → emission 0). Read by the composite as a surface
        //                           property (no voxel cross-contamination / flicker).
        //   @builtin(frag_depth)  : reverse-Z, unchanged.
        struct FragmentOutput {
            @location(0) albedo : vec4<f32>,
            @location(1) normal : vec4<f32>,
            @location(2) emission : vec4<f32>,
            @builtin(frag_depth) depth : f32,
        };

        ${sceneSDF}

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
            let hz = footprint_half_z(instance_index);
            let center = vec3<f32>(transform[3].x, transform[3].y, transform[3].z);

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
            let hz = footprint_half_z(instance_index);
            let center = vec3<f32>(transform[3].x, transform[3].y, transform[3].z);
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
                let d = sd_shape3d(lo + ld * t, instance_index);
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
            let nLocal = sd_normal3d(pLocal, instance_index);

            // Back to world (forward yaw).
            let fc = cos(yaw);
            let fs_ = sin(yaw);
            let nWorld = normalize(vec3<f32>(rotZ(nLocal.xy, fc, fs_), nLocal.z));
            let pWorld = center + vec3<f32>(rotZ(pLocal.xy, fc, fs_), pLocal.z);

            let clip = uViewProj * vec4<f32>(pWorld, 1.0);

            // G-buffer write. No fixed directional shade here anymore — the RC
            // composite supplies all lighting (omni + normal-aware directional).
            // uLightDir is now unused by fs_main (left declared for other passes).
            var out: FragmentOutput;
            out.albedo = vec4<f32>(color.rgb, 1.0);
            // World normal packed into [0,1]; a = 1 marks "surface present".
            out.normal = vec4<f32>(nWorld * 0.5 + 0.5, 1.0);
            // Per-pixel self-emission (surface property). SAME formula as
            // emission_of() in voxelize.shader.ts: pure occluder material.x==0 → 0.
            out.emission = vec4<f32>(color.rgb * abs(uMaterial[instance_index].x), 1.0);
            // Reverse-Z: viewProj already maps NEAR→1, FAR→0 to match the pipeline's
            // greater-equal/clear-0 depth compare. See ResizeSystem.viewProjMatrix.
            out.depth = clip.z / clip.w;
            return out;
        }

        // ============= Emission pass: emitter/occluder map for RC =============
        // Rasterizes the SAME per-instance impostor box as vs_main, then raymarches
        // the local 3D SDF to determine COVERAGE — so the occluder mask matches the
        // visible 2.5D silhouette under tilt. No frag_depth (RC is screen-space 2D).
        //
        // Attachment 0 (emissionTexture rgba16float, ADDITIVE):
        //   emitters  -> rgb = Color.rgb * abs(intensity) * edge (premultiplied HDR),
        //                a   = opacity * edge
        //   occluders -> rgb = 0, a = coverage (== edge)
        //   opacity = 1 - clamp(Translucency, 0, 1); edge = Blurness feather else 1.
        // Attachment 1 (emitDirTexture rg16float, REPLACE):
        //   omni (intensity >= 0)        -> (0, 0)
        //   directional (intensity < 0)  -> normalize(transform[0].xy), or
        //                                   transform[1].xy for trapezoid beams.

        struct EmitOutput {
            @location(0) color: vec4<f32>,
            @location(1) dir:   vec2<f32>,
        };

        // Edge feather: 1 deep inside the shape, ramping to 0 at the outline over
        // uMaterial.z (blurness) local units. 0 keeps the hard SDF edge.
        fn edge_softness(dist: f32, instance_index: u32) -> f32 {
            let blur = uMaterial[instance_index].z;
            if (blur <= 0.0) {
                return 1.0;
            }
            return clamp(-dist / blur, 0.0, 1.0);
        }

        @vertex
        fn vs_emit(
            @builtin(vertex_index) vertex_index: u32,
            @builtin(instance_index) instance_index: u32
        ) -> VertexOutput {
            // Identical impostor box to vs_main.
            let transform = uTransform[instance_index];
            let hz = footprint_half_z(instance_index);
            let center = vec3<f32>(transform[3].x, transform[3].y, transform[3].z);

            let yaw = atan2(transform[0].y, transform[0].x);

            let halfXY = footprint_half_xy(instance_index);
            let corner = CUBE[vertex_index];

            let scaled = corner * vec3<f32>(halfXY.x, halfXY.y, hz);
            let xy = rotZ(scaled.xy, cos(yaw), sin(yaw));
            let world = center + vec3<f32>(xy, scaled.z);

            var out: VertexOutput;
            out.position = uViewProj * vec4<f32>(world, 1.0);
            out.instance_index = instance_index;
            out.world = world;
            return out;
        }

        @fragment
        fn fs_emit(
            @builtin(position) frag_coord: vec4<f32>,
            @location(0) @interpolate(flat) instance_index: u32,
            @location(1) world: vec3<f32>,
        ) -> EmitOutput {
            let color = uColor[instance_index];
            if (color.a == 0.0) {
                discard;
            }

            let transform = uTransform[instance_index];
            let hz = footprint_half_z(instance_index);
            let center = vec3<f32>(transform[3].x, transform[3].y, transform[3].z);
            let yaw = atan2(transform[0].y, transform[0].x);

            // World ray -> instance-local space (same as fs_main).
            let ic = cos(-yaw);
            let is = sin(-yaw);
            let relW = world - center;
            let lo = vec3<f32>(rotZ(relW.xy, ic, is), relW.z);
            let ld = normalize(vec3<f32>(rotZ(uRayDir.xy, ic, is), uRayDir.z));

            // Slab test against the local AABB.
            let halfXY = footprint_half_xy(instance_index);
            let half = vec3<f32>(halfXY.x, halfXY.y, hz);
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

            // Sphere-trace for coverage. dist at the hit feeds the edge feather.
            var t = max(t0, 0.0);
            var hit = false;
            var hitDist = 1.0;
            for (var i = 0; i < 96; i = i + 1) {
                let d = sd_shape3d(lo + ld * t, instance_index);
                if (d < 0.001) {
                    hit = true;
                    hitDist = d;
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

            let intensity = uMaterial[instance_index].x;

            // Sign of intensity encodes the directional flag: negative = directional,
            // non-negative = omni (write (0,0)). Color always uses abs() so the flag
            // never darkens the emitter. Facing = the instance's world +X axis;
            // trapezoid emitters are beams and face along their long axis (+Y).
            var dir = vec2<f32>(0.0);
            if (intensity < 0.0) {
                // World-space facing: +X axis (long axis +Y for trapezoid beams).
                var facingWorld = vec2<f32>(transform[0].x, transform[0].y);
                if (uKind[instance_index] == 4u) {
                    facingWorld = vec2<f32>(transform[1].x, transform[1].y);
                }
                // Project to screen/texture space so the cone matches the cascade
                // raymarch (which dots emitDir against -rayDir in texture UV space).
                // Drop z, then negate clip.y: clip is y-up, texture V grows downward
                // (rayDir = vec2(cos, -sin)). Same handedness as the composite's nScreen.
                let facingClip = (uViewProj * vec4<f32>(facingWorld, 0.0, 0.0)).xy;
                dir = normalize(vec2<f32>(facingClip.x, -facingClip.y));
            }

            // opacity = per-material 1 - translucency; edge feathers the boundary.
            let opacity = 1.0 - clamp(uMaterial[instance_index].y, 0.0, 1.0);
            let edge = edge_softness(hitDist, instance_index);

            var out: EmitOutput;
            out.color = vec4<f32>(color.rgb * abs(intensity) * edge, opacity * edge);
            out.dir = dir;
            return out;
        }
    `,
);

// console.log('>>', shaderMeta.shader.trim());
