import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";
import { VoxelBakedConfig } from "./voxelConfig.ts";
import { probePackWGSL } from "./voxelProbeShared.wgsl.ts";

// VCT — SCREEN-SPACE PROBE gather. One thread per probe (uniform OR adaptive) traces a hemisphere of
// fill cones through the voxelRadiance pyramid, projects the gathered radiance onto SH-L1 (4 coeffs /
// channel), and stores it into three rgba16float 2D textures + the probe's representative pixel/
// validity into one rgba32float texture (4 storage textures = the WebGPU default cap).
//
// TWO PIPELINES from this ONE factory, bake IS_ADAPTIVE (0/1):
//   - the UNIFORM variant (IS_ADAPTIVE=0) covers slots [0, numUniform) via a DIRECT dispatch of
//     ceil(numUniform / GATHER_WG) workgroups. It runs BEFORE the refine passes so the refine can
//     read its raw SH (the light-adaptive subdivision signal).
//   - the ADAPTIVE variant (IS_ADAPTIVE=1) covers slots [numUniform, numUniform+adaptiveCount) via
//     dispatchWorkgroupsIndirect (PASS B built the args from the live counter), so the empty atlas
//     tail costs nothing. It runs AFTER refine placed the adaptive probes.
// Both write the SAME raw SH atlas (uniform → rows [0, gh), adaptive → rows [gh, ...)). The cone/SH
// math is byte-for-byte the flat-atlas gather's; ONLY the addressing changed:
//   - the thread indexes a GLOBAL slot (1D) → atlas texel (slot % gw, slot / gw);
//   - the representative pixel comes from probeData[slot] (written by classify/refine), not recomputed.
//
// STORAGE = RAW SH RADIANCE coefficients (solid-angle weighted, no cosine lobe). The cosine
// (irradiance) convolution is applied at RECONSTRUCTION time in the cone shader (sh_avg_radiance).
// NO temporal accumulation, NO history: probes are regenerated from scratch each frame from the
// live voxelRadiance. Noise is a non-issue because the gather is VCT cone tracing (prefiltered).

// COMPUTE-visibility group-0 uniform helper.
const uC = (name: string, type: string) =>
  new VariableMeta(name, VariableKind.Uniform, type, { visibility: GPUShaderStage.COMPUTE });

// One thread per probe, 64 threads / workgroup (matches the old 1-thread/probe budget). Also drives
// the uniform DIRECT dispatch (ceil(numUniform / GATHER_WG)) AND PASS B's ceil(adaptive / GATHER_WG)
// args math → exported so the CPU dispatch, the args shader, and this shader cannot disagree.
export const GATHER_WORKGROUP = 64;

// isAdaptive bakes IS_ADAPTIVE: false → the uniform variant (direct dispatch over [0, numUniform)),
// true → the adaptive variant (indirect dispatch over the live counter's [numUniform, ...) tail).
export function createScreenProbeShaderMeta(cfg: VoxelBakedConfig, isAdaptive: boolean) {
  return new ShaderMeta(
  {
    // ---- group 0 : uniforms (COMPUTE-only) ----
    // .xyz = world min corner of the grid box, .w = cellSize (world units per voxel).
    gridOrigin: uC("uGridOrigin", `vec4<f32>`),
    // .xyz = voxel counts per axis, .w unused.
    gridDims: uC("uGridDims", `vec4<i32>`),
    // inverse(viewProjMatrix) (reverse-Z) — reconstructs the probe's world position from the
    // G-buffer depth at its representative pixel.
    invViewProj: uC("uInvViewProj", `mat4x4<f32>`),
    // .x = canvas width (px), .y = canvas height (px), .z = SCREEN_PROBE_TILE (full-res px / probe),
    // .w = maxAdaptive (the adaptive budget → caps the counter when deriving `total`). Placement is
    // already resolved into probeData by the classify/refine passes; the gather only reads it.
    screenParams: uC("screenParams", `vec4<f32>`),

    // ---- group 0 : G-buffer + voxelRadiance pyramid (Texture/Sampler => group 0) ----
    depthTex: new VariableMeta("depthTex", VariableKind.Texture, `texture_depth_2d`, {
      visibility: GPUShaderStage.COMPUTE,
      textureSampleType: "depth",
    }),
    normalTex: new VariableMeta("normalTex", VariableKind.Texture, `texture_2d<f32>`, {
      visibility: GPUShaderStage.COMPUTE,
      textureSampleType: "float",
    }),
    voxelRadiance: new VariableMeta("voxelRadiance", VariableKind.Texture, `texture_3d<f32>`, {
      visibility: GPUShaderStage.COMPUTE,
      viewDimension: "3d",
      textureSampleType: "float",
    }),
    voxelSampler: new VariableMeta("voxelSampler", VariableKind.Sampler, `sampler`, {
      visibility: GPUShaderStage.COMPUTE,
    }),

    // ---- group 1 : probe indirection (StorageRead = var<storage, read>) ----
    // probeData drives the addressing (repr pixel per slot); probeCounter[0] = the adaptive count
    // (read as plain u32 across the barrier) → total = numUniform + min(counter, maxAdaptive).
    probeData: new VariableMeta("uProbeData", VariableKind.StorageRead, `array<vec4<u32>>`, {
      visibility: GPUShaderStage.COMPUTE,
    }),
    probeCounter: new VariableMeta("uCounter", VariableKind.StorageRead, `array<u32, 2>`, {
      visibility: GPUShaderStage.COMPUTE,
    }),

    // ---- group 2 : outputs (StorageTexture, write-only, 4 total = the WebGPU default cap) ----
    screenShR: new VariableMeta("screenShR", VariableKind.StorageTexture, `texture_storage_2d<rgba16float, write>`, {
      visibility: GPUShaderStage.COMPUTE,
      viewDimension: "2d",
      storageTextureFormat: "rgba16float",
      storageTextureAccess: "write-only",
    }),
    screenShG: new VariableMeta("screenShG", VariableKind.StorageTexture, `texture_storage_2d<rgba16float, write>`, {
      visibility: GPUShaderStage.COMPUTE,
      viewDimension: "2d",
      storageTextureFormat: "rgba16float",
      storageTextureAccess: "write-only",
    }),
    screenShB: new VariableMeta("screenShB", VariableKind.StorageTexture, `texture_storage_2d<rgba16float, write>`, {
      visibility: GPUShaderStage.COMPUTE,
      viewDimension: "2d",
      storageTextureFormat: "rgba16float",
      storageTextureAccess: "write-only",
    }),
    // .xy = representative full-res pixel coord, .z = validity (1 surface / 0 no surface), .w = the
    // probe FOOTPRINT (cell size in full-res px) → the resolve area-weights each probe by footprint².
    // rgba32float (point-loaded in the resolve) so the pixel coords survive without precision loss.
    screenProbePix: new VariableMeta("screenProbePix", VariableKind.StorageTexture, `texture_storage_2d<rgba32float, write>`, {
      visibility: GPUShaderStage.COMPUTE,
      viewDimension: "2d",
      storageTextureFormat: "rgba32float",
      storageTextureAccess: "write-only",
    }),
    // Per-probe world anchor P (.xyz) — the gather already reconstructs it for the cone origin, so
    // storing it lets the resolve skip the per-tap G-buffer unproject (P1). rgba32float for world-space
    // plane-reject precision.
    screenProbePos: new VariableMeta("screenProbePos", VariableKind.StorageTexture, `texture_storage_2d<rgba32float, write>`, {
      visibility: GPUShaderStage.COMPUTE,
      viewDimension: "2d",
      storageTextureFormat: "rgba32float",
      storageTextureAccess: "write-only",
    }),
    // Per-probe world normal N (.xyz, normalized). rgba16float half is ample for the resolve's
    // normal-similarity weight; paired with screenProbePos it replaces the per-tap normal+depth reads.
    screenProbeNrm: new VariableMeta("screenProbeNrm", VariableKind.StorageTexture, `texture_storage_2d<rgba16float, write>`, {
      visibility: GPUShaderStage.COMPUTE,
      viewDimension: "2d",
      storageTextureFormat: "rgba16float",
      storageTextureAccess: "write-only",
    }),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
const PI: f32 = 3.14159265359;
const IS_ADAPTIVE: i32 = ${isAdaptive ? 1 : 0};   // baked per pipeline (uniform=0 / adaptive=1)
const CONES_PER_PROBE: i32 = ${cfg.conesPerProbe};
const MAX_DIST: f32 = ${cfg.maxDist};
const APERTURE: f32 = ${cfg.aperture};
const NORMAL_BIAS: f32 = ${cfg.normalBias};

${probePackWGSL}

// One fill cone marched from origin along dir through the voxelRadiance pyramid: premultiplied
// front-to-back "over" integration (diameter grows with distance, LOD = log2(diameter/voxelSize),
// step floored at reach/64, early-out on full opacity / past reach / outside the box).
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

// Unproject an NDC point (z reverse-Z) to world space. Copied from voxelCone.shader.ts.
fn unproject(ndc: vec3<f32>) -> vec3<f32> {
  let w = uInvViewProj * vec4<f32>(ndc, 1.0);
  return w.xyz / w.w;
}

// Orthonormal basis with column 2 = n (basis * (x,y,z) = x*t + y*b + z*n).
fn build_basis(n: vec3<f32>) -> mat3x3<f32> {
  let a = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.9);
  let t = normalize(cross(a, n));
  let b = cross(n, t);
  return mat3x3<f32>(t, b, n);
}

@compute @workgroup_size(${GATHER_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // Atlas width in probes = gw = coarse-grid width; uniform block = numUniform slots.
  let gw = i32(ceil(screenParams.x / screenParams.z));
  let gh = i32(ceil(screenParams.y / screenParams.z));
  let numUniform = gw * gh;
  let maxAdaptive = i32(screenParams.w);
  // IS_ADAPTIVE (baked) selects which block this pipeline gathers. UNIFORM: base 0, count = numUniform
  // (a direct dispatch). ADAPTIVE: base numUniform, count = the (capped) live counter (indirect args).
  // BOTH branches of the select reference uCounter textually, so neither variant's binding is dropped
  // and the two explicit layouts stay identical. Threads past count were only launched by the
  // ceil-rounded group count (direct or indirect) — drop them.
  let base = select(0, numUniform, IS_ADAPTIVE == 1);
  let count = select(numUniform, min(i32(uCounter[0]), maxAdaptive), IS_ADAPTIVE == 1);
  let localIdx = i32(gid.x);
  if (localIdx >= count) { return; }
  let globalSlot = base + localIdx;

  // FLAT-ATLAS slot→texel: uniform slots (row-major identity) fill rows [0, gh); adaptive slots fill
  // rows [gh, gh+adaptiveRows). One texel per probe either way.
  let texel = vec2<i32>(globalSlot % gw, globalSlot / gw);

  // Representative full-res pixel + FOOTPRINT (probeData.z: cell size in full-res px, set by
  // classify/refine) from probeData (placed by the classify/refine passes). The footprint rides into
  // screenProbePix.w so the resolve can area-weight each probe by its footprint² (density-invariant
  // average — N fine probes weigh the same collectively as the 1 coarse probe they subdivide).
  let rec = uProbeData[globalSlot];
  let full = sp_unpack_pixel(rec);
  let footprint = f32(rec.z);

  // No surface at this pixel → mark the probe invalid (the resolve leans on neighbors) and zero its
  // SH. n.a<0.5 = the G-buffer has no geometry here. (footprint may be 0 here; the resolve skips
  // invalid probes anyway on the .z<0.5 validity test.)
  let n = textureLoad(normalTex, full, 0);
  if (n.a < 0.5) {
    textureStore(screenShR, texel, vec4<f32>(0.0));
    textureStore(screenShG, texel, vec4<f32>(0.0));
    textureStore(screenShB, texel, vec4<f32>(0.0));
    textureStore(screenProbePix, texel, vec4<f32>(f32(full.x), f32(full.y), 0.0, footprint));
    textureStore(screenProbePos, texel, vec4<f32>(0.0));
    textureStore(screenProbeNrm, texel, vec4<f32>(0.0));
    return;
  }

  // Reconstruct the probe anchor P + normal N from the G-buffer, then lift the cone origin off the
  // surface (SAME lift as voxelCone's per-pixel origin) to avoid self-sampling the surface voxel.
  let N = normalize(n.rgb * 2.0 - 1.0);
  let depth = textureLoad(depthTex, full, 0);
  let uv = (vec2<f32>(full) + vec2<f32>(0.5)) / screenParams.xy;
  let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth);
  let P = unproject(ndc);
  let origin = P + N * (uGridOrigin.w * 1.5 + NORMAL_BIAS);

  let C = CONES_PER_PROBE;
  let reach = MAX_DIST;
  let aperture = APERTURE;

  // SH-L1 radiance accumulators, per color channel (.xyzw = L00, L1m1, L10, L11).
  var cR = vec4<f32>(0.0);
  var cG = vec4<f32>(0.0);
  var cB = vec4<f32>(0.0);

  // Trace C fill cones over the UPPER HEMISPHERE around N (Fibonacci hemisphere). A screen probe
  // sits ON a surface, so only the upper hemisphere carries incoming light. dw = 2*PI/C.
  let nBasis = build_basis(N);
  let dw = 2.0 * PI / f32(C);
  for (var i = 0; i < C; i = i + 1) {
    let cosT = 1.0 - (f32(i) + 0.5) / f32(C);   // [0,1) → upper hemisphere (cosT = component along N)
    let sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
    let phi = f32(i) * 2.39996323;              // golden angle
    let dir = nBasis * vec3<f32>(sinT * cos(phi), sinT * sin(phi), cosT); // local hemisphere → world

    let r = trace_probe_cone(origin, dir, aperture, reach);

    // Real SH-L1 basis evaluated at the WORLD dir.
    let Y00 = 0.282095;
    let Y1m1 = 0.488603 * dir.y;
    let Y10 = 0.488603 * dir.z;
    let Y11 = 0.488603 * dir.x;
    let shb = vec4<f32>(Y00, Y1m1, Y10, Y11) * dw;
    cR = cR + r.r * shb;
    cG = cG + r.g * shb;
    cB = cB + r.b * shb;
  }

  textureStore(screenShR, texel, cR);
  textureStore(screenShG, texel, cG);
  textureStore(screenShB, texel, cB);
  // Valid probe: store the representative full-res pixel + validity 1 + footprint (.w) for the
  // bilateral resolve (the resolve area-weights each probe by footprint²).
  textureStore(screenProbePix, texel, vec4<f32>(f32(full.x), f32(full.y), 1.0, footprint));
  // Store the reconstructed world anchor P + normal N so the resolve reads them directly (P1) instead
  // of re-doing normal+depth+unproject per tap.
  textureStore(screenProbePos, texel, vec4<f32>(P, 0.0));
  textureStore(screenProbeNrm, texel, vec4<f32>(N, 0.0));
}
`,
  );
}
