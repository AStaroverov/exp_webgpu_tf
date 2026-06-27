import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";

// VCT — IRRADIANCE PROBE VOLUME (one compute pass, one thread per probe). This moves the
// expensive low-frequency FILL hemisphere out of the per-pixel cone pass into a LOW-RES 3D
// volume computed ONCE per frame: every probe traces a full SPHERE of fill cones through the
// voxelRadiance pyramid, projects the gathered radiance onto SH-L1 (4 coeffs / channel), and
// stores it into three rgba16float 3D textures (one per color channel, .xyzw = L00,L1m1,L10,L11).
//
// WHY: the per-pixel fill term cost scaled with screen resolution AND object count. A probe
// volume decouples it from BOTH — fill GI is now O(probes·cones) regardless of pixels or scene
// size, sampled per-pixel with one hardware-trilinear fetch. The probe box is IDENTICAL to the
// voxel grid box (same origin + extent); only the resolution differs (probeDims << gridDims).
// Re-computed every frame from the live voxel volume → no temporal accumulation, no ghosting.
//
// The cone integration here MIRRORS trace_cone() in voxelCone.shader.ts exactly (premultiplied
// front-to-back "over", diameter grows with distance, LOD = log2(diameter/voxelSize)), so the
// probe fill matches what the old per-pixel fill cones gathered. We DO NOT import the cone file
// (its trace_cone reads the cone shader's own uniforms); the march is inlined against THIS pass's
// uniforms instead.
//
// STORAGE = RAW SH RADIANCE coefficients (solid-angle weighted, no cosine lobe). The cosine
// (irradiance) convolution is applied at RECONSTRUCTION time in the cone shader (sh_irradiance),
// so the probe stays a pure radiance-SH representation that any reconstruction can re-weight.

// COMPUTE-visibility group-0 uniform helper (mirrors voxelize.shader.ts / voxelMip.shader.ts).
const uC = (name: string, type: string) =>
  new VariableMeta(name, VariableKind.Uniform, type, { visibility: GPUShaderStage.COMPUTE });

export const WORKGROUP = 4; // 4*4*4 = 64 threads/workgroup over the 3D probe grid

export const shaderMeta = new ShaderMeta(
  {
    // ---- group 0 : uniforms (COMPUTE-only) ----
    // .xyz = world min corner of the grid box, .w = cellSize (world units per voxel). SAME box
    // as the cone/voxelize grid → extent = gridDims.xyz * cellSize.
    gridOrigin: uC("uGridOrigin", `vec4<f32>`),
    // .xyz = voxel counts per axis (defines the world extent together with cellSize), .w unused.
    gridDims: uC("uGridDims", `vec4<i32>`),
    // .xyz = probe counts per axis (the resolution of THIS volume), .w unused.
    probeDims: uC("uProbeDims", `vec4<i32>`),
    // .x = cones-per-probe (read as i32), .y = maxDist / cone reach (world units), .z = aperture
    // = tan(halfAngle), .w spare.
    probeParams: uC("uProbeParams", `vec4<f32>`),

    // ---- group 0 : voxelRadiance pyramid (Texture => @group(0)) + linear sampler ----
    // Declared as Texture/Sampler so they land in group 0 alongside the uniforms (mapKindToGroup),
    // keeping group 2 free for the SH storage outputs (StorageTexture => group 2).
    voxelRadiance: new VariableMeta("voxelRadiance", VariableKind.Texture, `texture_3d<f32>`, {
      visibility: GPUShaderStage.COMPUTE,
      viewDimension: "3d",
      textureSampleType: "float",
    }),
    voxelSampler: new VariableMeta("voxelSampler", VariableKind.Sampler, `sampler`, {
      visibility: GPUShaderStage.COMPUTE,
    }),

    // ---- group 2 : SH-L1 outputs (StorageTexture, write-only, one per color channel) ----
    // .xyzw = the 4 SH-L1 coefficients (L00, L1m1, L10, L11) of that channel's RADIANCE.
    shR: new VariableMeta("shR", VariableKind.StorageTexture, `texture_storage_3d<rgba16float, write>`, {
      visibility: GPUShaderStage.COMPUTE,
      viewDimension: "3d",
      storageTextureFormat: "rgba16float",
      storageTextureAccess: "write-only",
    }),
    shG: new VariableMeta("shG", VariableKind.StorageTexture, `texture_storage_3d<rgba16float, write>`, {
      visibility: GPUShaderStage.COMPUTE,
      viewDimension: "3d",
      storageTextureFormat: "rgba16float",
      storageTextureAccess: "write-only",
    }),
    shB: new VariableMeta("shB", VariableKind.StorageTexture, `texture_storage_3d<rgba16float, write>`, {
      visibility: GPUShaderStage.COMPUTE,
      viewDimension: "3d",
      storageTextureFormat: "rgba16float",
      storageTextureAccess: "write-only",
    }),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
const PI: f32 = 3.14159265359;

// One fill cone marched from origin along dir through the voxelRadiance pyramid — IDENTICAL
// premultiplied front-to-back "over" integration as trace_cone() in voxelCone.shader.ts
// (diameter grows with distance, LOD = log2(diameter/voxelSize), step floored at reach/64,
// early-out on full opacity / past reach / outside the box). Returns the gathered radiance +
// accumulated opacity (.a). No fade window: probe fill cones reach the global maxDist and we
// only consume their .rgb (the SH projection below).
fn trace_probe_cone(origin: vec3<f32>, dir: vec3<f32>, aperture: f32, reach: f32) -> vec4<f32> {
  var col = vec3<f32>(0.0);
  var alpha = 0.0;
  let voxelSize = uGridOrigin.w;
  let gridMin = uGridOrigin.xyz;
  let extent = vec3<f32>(uGridDims.xyz) * voxelSize;
  let stepFloor = reach / 64.0;
  var dist = voxelSize;
  for (var i = 0; i < 64; i = i + 1) {
    if (alpha >= 1.0 || dist > reach) { break; }
    let diameter = max(voxelSize, 2.0 * aperture * dist);
    let lod = log2(diameter / voxelSize);
    let wp = origin + dir * dist;
    let uvw = (wp - gridMin) / extent;
    if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) { break; }
    let s = textureSampleLevel(voxelRadiance, voxelSampler, uvw, lod);
    col = col + (1.0 - alpha) * s.rgb;
    alpha = alpha + (1.0 - alpha) * s.a;
    dist = dist + max(diameter * 0.5, stepFloor);
  }
  return vec4<f32>(col, alpha);
}

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let coord = vec3<i32>(gid);
  // Dispatch is ceil-rounded; drop threads past the probe-grid bounds.
  if (coord.x >= uProbeDims.x || coord.y >= uProbeDims.y || coord.z >= uProbeDims.z) {
    return;
  }

  // Probe center in world space. The probe grid spans the SAME box as the voxel grid; a probe
  // sits at the center of its cell → (coord + 0.5) / probeDims maps to [0,1] across the box.
  let boxMin = uGridOrigin.xyz;
  let extent = vec3<f32>(uGridDims.xyz) * uGridOrigin.w;
  let P = boxMin + (vec3<f32>(coord) + vec3<f32>(0.5)) / vec3<f32>(uProbeDims.xyz) * extent;

  let C = max(1, i32(uProbeParams.x));
  let reach = uProbeParams.y;
  let aperture = uProbeParams.z;

  // SH-L1 radiance accumulators, per color channel (.xyzw = L00, L1m1, L10, L11).
  var cR = vec4<f32>(0.0);
  var cG = vec4<f32>(0.0);
  var cB = vec4<f32>(0.0);

  // Trace C fill cones over the FULL SPHERE (Fibonacci sphere — NOT a hemisphere; a probe has
  // no surface normal, it integrates incoming radiance from every direction). Project each
  // cone's gathered radiance onto SH-L1 with the per-cone solid-angle weight 4*PI/C.
  let dw = 4.0 * PI / f32(C);
  for (var i = 0; i < C; i = i + 1) {
    let cosT = 1.0 - 2.0 * (f32(i) + 0.5) / f32(C);
    let sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
    let phi = f32(i) * 2.39996323;            // golden angle
    let dir = vec3<f32>(sinT * cos(phi), sinT * sin(phi), cosT);

    let r = trace_probe_cone(P, dir, aperture, reach);

    // Real SH-L1 basis evaluated at dir.
    let Y00 = 0.282095;
    let Y1m1 = 0.488603 * dir.y;
    let Y10 = 0.488603 * dir.z;
    let Y11 = 0.488603 * dir.x;
    let basis = vec4<f32>(Y00, Y1m1, Y10, Y11) * dw;
    cR = cR + r.r * basis;
    cG = cG + r.g * basis;
    cB = cB + r.b * basis;
  }

  textureStore(shR, coord, cR);
  textureStore(shG, coord, cG);
  textureStore(shB, coord, cB);
}
`,
);
