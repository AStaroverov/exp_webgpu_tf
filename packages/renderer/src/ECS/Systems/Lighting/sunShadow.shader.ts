import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";
import { MAX_INSTANCE_COUNT } from "../SDFSystem/sdf.shader.ts";
import { sceneSDF } from "../SDFSystem/sceneSDF.wgsl.ts";

// Sun shadow-map depth pass. Renders the SDF scene from the sun's point of view into a
// single depth texture; the composite later projects each lit pixel into this map and
// compares depths to decide if it is shadowed. This gives crisp SDF silhouettes (vs the
// blocky voxel-occupancy march it replaces).
//
// Mirrors sdf.shader.ts: one instanced draw of a 36-vertex impostor cube, rasterized
// through the SUN view-projection, then a 96-step sphere-trace of the instance-local 3D
// SDF along the sun's TRAVEL direction (uRayDir = -dirTowardSun). On a hit we write the
// hit point's sun-clip depth to @builtin(frag_depth); a miss discards. NO color outputs.
//
// DEPTH CONVENTION — STANDARD (NOT reverse-Z). The sun pipeline uses orthoZO (z in [0,1],
// near→0 far→1) with depthCompare "less-equal" + depthClearValue 1.0 → the map keeps the
// SMALLEST (nearest-to-sun) depth per texel. This is deliberately decoupled from the main
// camera's reverse-Z so the standard shadow-map test is the simple one.
//
// group0 = uViewProj (sun) + uRayDir (sun travel dir). group1 = the 7 scene-instance
// StorageRead buffers, IDENTICAL names/types/order to voxelize.shader.ts / sdf.shader.ts
// so the shared ${sceneSDF} helpers resolve uKind/uValues/uRoundness/uTransform
// by global name, and so the bind-group numbering lines up 1:1 with sceneInstances.*.
// These are a RENDER pass (vertex+fragment), so the scene buffers use the DEFAULT
// VERTEX|FRAGMENT visibility (NOT COMPUTE like voxelize).

export const shaderMeta = new ShaderMeta(
  {
    // ---- group 0 : uniforms ----
    // Sun orthographic view-projection (orthoZO * lookAt). Maps world → sun clip, z in [0,1].
    viewProj: new VariableMeta("uViewProj", VariableKind.Uniform, `mat4x4<f32>`),
    // xyz = sun TRAVEL direction (= -dirTowardSun), normalized; w unused. The march dir,
    // analogous to sdf.shader's uRayDir (camera forward).
    rayDir: new VariableMeta("uRayDir", VariableKind.Uniform, `vec4<f32>`),

    // ---- group 1 : per-instance scene storage (StorageRead => @group(1)) ----
    // IDENTICAL global names + WGSL types + DECLARATION ORDER to voxelize.shader.ts so the
    // ${sceneSDF} helpers read uKind/uValues/uRoundness/uTransform by global name and
    // bindings 0..5 line up with sceneInstances.* in the createVoxelSystem sun group1 build.
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
    material: new VariableMeta(
      "uMaterial",
      VariableKind.StorageRead,
      `array<vec4<f32>, ${MAX_INSTANCE_COUNT}>`,
    ),
  },
  {},
  wgsl /* WGSL */ `
        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) @interpolate(flat) instance_index: u32,
            @location(1) world: vec3<f32>,
        };

        ${sceneSDF}

        // 36-vertex unit cube ([-1,1]^3), 12 triangles. Same impostor as sdf.shader.ts.
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

            let yaw = atan2(transform[0].y, transform[0].x);

            let halfXY = footprint_half_xy(instance_index);
            let corner = CUBE[vertex_index];

            let scaled = corner * vec3<f32>(halfXY.x, halfXY.y, hz);
            let xy = rotZ(scaled.xy, cos(yaw), sin(yaw));
            let world = center + vec3<f32>(xy, scaled.z);

            var out: VertexOutput;
            // Rasterize through the SUN view-projection (uViewProj is the sun matrix here).
            out.position = uViewProj * vec4<f32>(world, 1.0);
            out.instance_index = instance_index;
            out.world = world;
            return out;
        }

        // Sphere-trace the instance-local 3D SDF along the sun travel ray; write the hit's
        // sun-clip depth. Discard on miss / empty instance. Depth-only (no color targets).
        @fragment
        fn fs_depth(
            @builtin(position) frag_coord: vec4<f32>,
            @location(0) @interpolate(flat) instance_index: u32,
            @location(1) world: vec3<f32>,
        ) -> @builtin(frag_depth) f32 {
            let color = uColor[instance_index];
            if (color.a == 0.0) {
                discard;
            }

            let transform = uTransform[instance_index];
            let hz = footprint_half_z(instance_index);
            let center = vec3<f32>(transform[3].x, transform[3].y, transform[3].z);
            let yaw = atan2(transform[0].y, transform[0].x);

            // World ray (origin = this fragment's box-surface point) → instance-local space
            // (remove yaw). The ray direction is the SUN travel direction (uRayDir).
            let ic = cos(-yaw);
            let is = sin(-yaw);
            let relW = world - center;
            let lo = vec3<f32>(rotZ(relW.xy, ic, is), relW.z);
            let ld = normalize(vec3<f32>(rotZ(uRayDir.xy, ic, is), uRayDir.z));

            // Slab test against the local AABB (footprint half-extents + half-height).
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

            // Back to world (forward yaw), then to sun clip space.
            let fc = cos(yaw);
            let fs_ = sin(yaw);
            let pWorld = center + vec3<f32>(rotZ(pLocal.xy, fc, fs_), pLocal.z);
            let clip = uViewProj * vec4<f32>(pWorld, 1.0);

            // Standard depth (orthoZO, z in [0,1]); the pipeline compares "less-equal".
            return clip.z / clip.w;
        }
    `,
);
