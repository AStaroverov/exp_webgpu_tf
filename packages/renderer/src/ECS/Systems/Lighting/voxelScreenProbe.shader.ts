import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";
import { VoxelBakedConfig } from "./voxelConfig.ts";
import { probePackWGSL, probeWeightWGSL } from "./voxelProbeShared.wgsl.ts";
import { buildBasisWGSL, unprojectWGSL } from "./voxelTrace.wgsl.ts";

// VCT — SCREEN-SPACE PROBE gather. One thread per probe (uniform OR adaptive) traces a hemisphere of
// fill cones through the voxelRadiance pyramid, projects the gathered radiance onto SH-L1 (4 coeffs /
// channel), and stores it into three rgba16float 2D textures + the probe's representative pixel/
// validity, world anchor P, and world normal N (6 storage textures — above the WebGPU default cap
// of 4, so gpu.ts requests the adapter's maxStorageTexturesPerShaderStage).
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
//
// STAGE 3 — TEMPORAL ACCUMULATION on the probe atlas. The system ping-pongs TWO full atlas sets by
// frame parity: this pass storage-writes the CURRENT set (group 2) and SAMPLES last frame's set as
// HISTORY (hist* bindings, group 0 — read-only history needs no storage access, so the group-2
// storage-texture budget is untouched). The fresh anchor P is reprojected through the PREVIOUS
// frame's forward viewProj into the prev UNIFORM tile (adaptive slots are a live-counter allocation
// order with no frame-to-frame identity — the uniform rows [0, gh) are the persistent backbone, so
// adaptive probes too fall back to the uniform history tile covering their reprojected position),
// validated with the SHARED sp_plane_normal_weight (disocclusion ⇒ weight ≈ 0 ⇒ fresh only), and
// blended: sh = mix(fresh, history, hysteresis · weight). SH is LINEAR in its coefficients, so a
// plain lerp is a valid radiance blend. hysteresis = 0 (uTemporalParams.x) disables the whole path
// — byte-identical output to the pre-temporal build (the parity/rollback gate).
//
// PROBE-CENTRIC FINAL GATHER (the accepted architecture — see ./README.md):
// this gather owns ALL cone tracing. Besides the fill hemisphere it traces the AIMED emitter cones
// (one narrow cone per emitter, once per PROBE instead of once per pixel — the deleted per-pixel
// loop in voxelCone was O(pixels × lights) and the dominant frame cost) and binds the 6 anisotropic
// far-field volumes (the anti-leak matters exactly where LONG cones are traced, and that is here).
// The emitter light is projected into the SAME SH-L1 the fill cones write, so the cone pass's
// resolve carries it for free.

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
      // STAGE 3 (temporal): LAST frame's FORWARD viewProjMatrix (NOT an inverse — the CPU snapshots
      // the raw matrix after each frame's uploads). Projects the fresh world anchor P into the
      // previous frame's clip space → prev NDC → prev pixel → prev uniform tile = the history texel.
      prevViewProj: uC("uPrevViewProj", `mat4x4<f32>`),
      // STAGE 3 (temporal), all LIVE (per-frame uploads, no rebuild):
      //   .x = hysteresis (0..1). 0 = temporal OFF — the fresh-only path, byte-identical output.
      //   .y = frame index (mod 1024 on the CPU for f32 exactness) — drives the per-frame
      //        golden-angle rotation of the Fibonacci fill-cone set.
      //   .z = plane-reject threshold in WORLD units (spPlaneK × voxel cellSize — the same live GUI
      //        knob the resolve's plane test scales by, so one knob tunes both).
      //   .w = normal-similarity power (spNormalPow — the resolve's live normal weight, reused).
      temporalParams: uC("uTemporalParams", `vec4<f32>`),
      // .x = global live light count (DEBUG only — the aimed loop reads its cluster cell's list
      // from uLightClusters, group 1), .y = anisoMode (0 = isotropic pyramid, 1 = anisotropic
      // directional volumes), .zw spare.
      lightParams: uC("uLightParams", `vec4<f32>`),
      // The 6 ANISOTROPIC directional radiance volumes (−X,+X,−Y,+Y,−Z,+Z) — ALL-mips sampled
      // views of the half-res directional pyramid (voxelAnisoBase/voxelAnisoVolume). sample_aniso
      // picks the 3 facing the cone dir and blends by dir² → direction-correct occlusion for the
      // far-field (coarse-LOD) samples; the near field still reads voxelRadiance mip 0 (crisp).
      anisoNegX: new VariableMeta("anisoNegX", VariableKind.Texture, `texture_3d<f32>`, {
        visibility: GPUShaderStage.COMPUTE,
        viewDimension: "3d",
        textureSampleType: "float",
      }),
      anisoPosX: new VariableMeta("anisoPosX", VariableKind.Texture, `texture_3d<f32>`, {
        visibility: GPUShaderStage.COMPUTE,
        viewDimension: "3d",
        textureSampleType: "float",
      }),
      anisoNegY: new VariableMeta("anisoNegY", VariableKind.Texture, `texture_3d<f32>`, {
        visibility: GPUShaderStage.COMPUTE,
        viewDimension: "3d",
        textureSampleType: "float",
      }),
      anisoPosY: new VariableMeta("anisoPosY", VariableKind.Texture, `texture_3d<f32>`, {
        visibility: GPUShaderStage.COMPUTE,
        viewDimension: "3d",
        textureSampleType: "float",
      }),
      anisoNegZ: new VariableMeta("anisoNegZ", VariableKind.Texture, `texture_3d<f32>`, {
        visibility: GPUShaderStage.COMPUTE,
        viewDimension: "3d",
        textureSampleType: "float",
      }),
      anisoPosZ: new VariableMeta("anisoPosZ", VariableKind.Texture, `texture_3d<f32>`, {
        visibility: GPUShaderStage.COMPUTE,
        viewDimension: "3d",
        textureSampleType: "float",
      }),

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

      // ---- group 0 : STAGE-3 HISTORY (last frame's atlas set, SAMPLED — point textureLoad only) ----
      // The ping-pong partner of the group-2 outputs: what THIS pass wrote last frame. Bound as plain
      // sampled textures (TEXTURE_BINDING) so the group-2 storage-texture budget stays at 6. History
      // VALIDATION needs only pos/nrm: an invalid probe stores ZERO pos/nrm, so |nrm|² ≈ 0 doubles as
      // the history-validity test (freshly recreated textures are zero-filled → auto-invalid history
      // after a resize/tile change, no CPU flag needed). Sample types mirror the cone pass's bindings
      // of the same textures (rgba16float = float, rgba32float = unfilterable-float).
      histShR: new VariableMeta("histShR", VariableKind.Texture, `texture_2d<f32>`, {
        visibility: GPUShaderStage.COMPUTE,
        textureSampleType: "float",
      }),
      histShG: new VariableMeta("histShG", VariableKind.Texture, `texture_2d<f32>`, {
        visibility: GPUShaderStage.COMPUTE,
        textureSampleType: "float",
      }),
      histShB: new VariableMeta("histShB", VariableKind.Texture, `texture_2d<f32>`, {
        visibility: GPUShaderStage.COMPUTE,
        textureSampleType: "float",
      }),
      histPos: new VariableMeta("histPos", VariableKind.Texture, `texture_2d<f32>`, {
        visibility: GPUShaderStage.COMPUTE,
        textureSampleType: "unfilterable-float",
      }),
      histNrm: new VariableMeta("histNrm", VariableKind.Texture, `texture_2d<f32>`, {
        visibility: GPUShaderStage.COMPUTE,
        textureSampleType: "float",
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
      // Emitter records the aimed cones importance-sample, TWO vec4 per light (stride 2):
      //   [2j]   = .xyz world CENTER, .w radius (penumbra source / falloff scale)
      //   [2j+1] = .rgb emitter color, .w intensity → Lj = rgb·|w| is the true radiance (drives the
      //            analytic direct + bleed-cancel term — a blocked cone DARKENS instead of picking up
      //            the occluder's own emission = the "white shadow" bug; a BRIGHT occluder cancels
      //            its own false shadow).
      // A RUNTIME-SIZED storage array (no light-count cap — the old `array<vec4, 8>` uniform pair);
      // AUTO-DISCOVERED from the LightEmitter component, live count rides uLightParams.x, and the
      // aimed loop importance-samples AIMED_PER_FRAME of them per probe per frame (dominant lights
      // every frame; the temporal history integrates the stochastic dim tail).
      lightsData: new VariableMeta("uLights", VariableKind.StorageRead, `array<vec4<f32>>`, {
        visibility: GPUShaderStage.COMPUTE,
      }),
      // CLUSTERED LIGHT CULLING (Persson-style, CPU assignment): the grid AABB is divided into
      // cluster cells of CLUSTER_DIV voxels per axis; setLights bins each emitter into every cell its
      // influence sphere (derived from the same 0.003 contribution cull) overlaps. Layout: one
      // (CLUSTER_CAP + 1)-u32 record per cell — [base] = live count, [base + 1 + k] = uLights index.
      // The aimed loop reads ONLY its probe's cell list, so the per-probe cost tracks the LOCAL
      // light density, not the global count.
      lightClusters: new VariableMeta("uLightClusters", VariableKind.StorageRead, `array<u32>`, {
        visibility: GPUShaderStage.COMPUTE,
      }),

      // ---- group 2 : outputs (StorageTexture, write-only, 6 total — gpu.ts requests the adapter's
      // maxStorageTexturesPerShaderStage, above the WebGPU default cap of 4) ----
      screenShR: new VariableMeta(
        "screenShR",
        VariableKind.StorageTexture,
        `texture_storage_2d<rgba16float, write>`,
        {
          visibility: GPUShaderStage.COMPUTE,
          viewDimension: "2d",
          storageTextureFormat: "rgba16float",
          storageTextureAccess: "write-only",
        },
      ),
      screenShG: new VariableMeta(
        "screenShG",
        VariableKind.StorageTexture,
        `texture_storage_2d<rgba16float, write>`,
        {
          visibility: GPUShaderStage.COMPUTE,
          viewDimension: "2d",
          storageTextureFormat: "rgba16float",
          storageTextureAccess: "write-only",
        },
      ),
      screenShB: new VariableMeta(
        "screenShB",
        VariableKind.StorageTexture,
        `texture_storage_2d<rgba16float, write>`,
        {
          visibility: GPUShaderStage.COMPUTE,
          viewDimension: "2d",
          storageTextureFormat: "rgba16float",
          storageTextureAccess: "write-only",
        },
      ),
      // .xy = representative full-res pixel coord, .z = validity (1 surface / 0 no surface), .w = the
      // probe FOOTPRINT (cell size in full-res px) → the resolve area-weights each probe by footprint².
      // rgba32float (point-loaded in the resolve) so the pixel coords survive without precision loss.
      screenProbePix: new VariableMeta(
        "screenProbePix",
        VariableKind.StorageTexture,
        `texture_storage_2d<rgba32float, write>`,
        {
          visibility: GPUShaderStage.COMPUTE,
          viewDimension: "2d",
          storageTextureFormat: "rgba32float",
          storageTextureAccess: "write-only",
        },
      ),
      // Per-probe world anchor P (.xyz) — the gather already reconstructs it for the cone origin, so
      // storing it lets the resolve skip the per-tap G-buffer unproject (P1). rgba32float for world-space
      // plane-reject precision.
      screenProbePos: new VariableMeta(
        "screenProbePos",
        VariableKind.StorageTexture,
        `texture_storage_2d<rgba32float, write>`,
        {
          visibility: GPUShaderStage.COMPUTE,
          viewDimension: "2d",
          storageTextureFormat: "rgba32float",
          storageTextureAccess: "write-only",
        },
      ),
      // Per-probe world normal N (.xyz, normalized). rgba16float half is ample for the resolve's
      // normal-similarity weight; paired with screenProbePos it replaces the per-tap normal+depth reads.
      screenProbeNrm: new VariableMeta(
        "screenProbeNrm",
        VariableKind.StorageTexture,
        `texture_storage_2d<rgba16float, write>`,
        {
          visibility: GPUShaderStage.COMPUTE,
          viewDimension: "2d",
          storageTextureFormat: "rgba16float",
          storageTextureAccess: "write-only",
        },
      ),
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
// AIMED-emitter baked consts (see voxelConfig.ts).
const EMITTER_FALLOFF: f32 = ${cfg.emitterFalloff};
const EMITTER_DIRECT: f32 = ${cfg.emitterDirect};
const AIMED_STEPS: i32 = ${cfg.aimedSteps};
const AIMED_ALPHA_CUT: f32 = ${cfg.aimedAlphaCut};
// Per-probe per-frame aimed-cone budget. With ≤ AIMED_PER_FRAME live lights every one is traced
// every frame (the exact path). With more, each probe IMPORTANCE-SAMPLES this many lights per
// frame (systematic resampling over the unshadowed-contribution weights — see the aimed loop), so
// the aimed cost is CONSTANT in the light count while dominant lights are traced every frame.
const AIMED_PER_FRAME: i32 = ${cfg.aimedPerFrame};
// Cluster grid: CLUSTER_DIV voxels per cluster cell per axis; CLUSTER_CAP lights per cell record
// (must match the CPU binning in createVoxelSystem.setLights — both bake from the same config).
const CLUSTER_DIV: i32 = ${cfg.clusterDiv};
const CLUSTER_CAP: i32 = ${cfg.clusterCap};
// SH-L1 projection weight of one aimed (delta-direction) cone = 4π/3. NOT the fill cones'
// dw = 2π/C (emitter energy must not scale with CONES_PER_PROBE) and NOT the light's solid angle
// Ω = π·(r/d)² (the analytic direct term is already an INTEGRATED, irradiance-scale quantity —
// atten carries the distance falloff — so an Ω energy weight would apply the falloff twice; Ω only
// sets the trace APERTURE = penumbra width). 4π/3 is the weight for which sh_avg_radiance
// reconstructs ≈ ndl·L at ndl = 1, i.e. the old per-pixel scale (within SH-L1's softening).
const W_AIMED: f32 = 4.18879020;

${probePackWGSL}
${probeWeightWGSL}
${unprojectWGSL}
${buildBasisWGSL}

// Sample the 6 ANISOTROPIC directional volumes for a cone traveling along unit dir, at aniso LOD.
// Per axis pick the volume whose pre-integration faces the ray (sign of dir): the negX volume is
// accumulated front-to-back for a ray heading toward −X (built from the +X side), so dir.x<0 reads
// negX, dir.x>0 reads posX (same for Y/Z). Blend the 3 by dir² (Σ dir² = 1 for a unit dir) → the
// direction-correct occluded radiance.
fn sample_aniso(uvw: vec3<f32>, lod: f32, dir: vec3<f32>) -> vec4<f32> {
  let w = dir * dir;
  var sx: vec4<f32>;
  if (dir.x < 0.0) { sx = textureSampleLevel(anisoNegX, voxelSampler, uvw, lod); }
  else { sx = textureSampleLevel(anisoPosX, voxelSampler, uvw, lod); }
  var sy: vec4<f32>;
  if (dir.y < 0.0) { sy = textureSampleLevel(anisoNegY, voxelSampler, uvw, lod); }
  else { sy = textureSampleLevel(anisoPosY, voxelSampler, uvw, lod); }
  var sz: vec4<f32>;
  if (dir.z < 0.0) { sz = textureSampleLevel(anisoNegZ, voxelSampler, uvw, lod); }
  else { sz = textureSampleLevel(anisoPosZ, voxelSampler, uvw, lod); }
  return w.x * sx + w.y * sy + w.z * sz;
}

// The radiance fetch inside the cone march. uLightParams.y (live) toggles the model:
//   iso  (0): the plain isotropic voxelRadiance pyramid (one direction-agnostic average).
//   aniso(1): near field (lod≲1) reads the crisp iso mip 0; the far field reads the directional
//             volumes at (lod-1) — aniso level 0 == iso mip 1 resolution — blended over lod∈[0,1]
//             so there is no seam where the two representations meet.
// Both volumes store PREMULTIPLIED rgb (radiance·coverage) + coverage in .a, so the caller's
// front-to-back "over" operator is unchanged regardless of which model is active.
fn sample_radiance(uvw: vec3<f32>, lod: f32, dir: vec3<f32>) -> vec4<f32> {
  if (uLightParams.y < 0.5) {
    return textureSampleLevel(voxelRadiance, voxelSampler, uvw, lod);
  }
  let iso0 = textureSampleLevel(voxelRadiance, voxelSampler, uvw, min(lod, 1.0));
  let aniso = sample_aniso(uvw, max(0.0, lod - 1.0), dir);
  return mix(iso0, aniso, clamp(lod, 0.0, 1.0));
}

// One cone marched from origin along dir through the voxelRadiance pyramid: premultiplied
// front-to-back "over" integration (diameter grows with distance, LOD = log2(diameter/voxelSize),
// step floored at reach/maxSteps, early-out on alpha >= alphaCut / past reach / outside the box).
// Fill cones pass (64, 1.0) — the original fixed budget; aimed emitter cones pass the baked
// (AIMED_STEPS, AIMED_ALPHA_CUT) so their march cost stays tunable like the old per-pixel path.
fn trace_probe_cone(origin: vec3<f32>, dir: vec3<f32>, aperture: f32, reach: f32, maxSteps: i32, alphaCut: f32) -> vec4<f32> {
  var col = vec3<f32>(0.0);
  var alpha = 0.0;
  let voxelSize = uGridOrigin.w;
  let gridMin = uGridOrigin.xyz;
  let extent = vec3<f32>(uGridDims.xyz) * voxelSize;
  let stepFloor = reach / f32(maxSteps);
  var dist = voxelSize;
  for (var i = 0; i < maxSteps; i = i + 1) {
    if (alpha >= alphaCut || dist > reach) { break; }
    let diameter = max(voxelSize, 2.0 * aperture * dist);
    let lod = log2(diameter / voxelSize);
    let wp = origin + dir * dist;
    let uvw = (wp - gridMin) / extent;
    if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) { break; }
    let s = sample_radiance(uvw, lod, dir);
    col = col + (1.0 - alpha) * s.rgb;
    alpha = alpha + (1.0 - alpha) * s.a;
    dist = dist + max(diameter * 0.5, stepFloor);
  }
  return vec4<f32>(col, alpha);
}

// SHADOW cone for the aimed emitter lights — modeled on the reference traceShadowCone
// (Friduric/voxel-cone-tracing): a NEAR-UNIFORM ~voxel-sized march with saturating front-to-back
// accumulation, NOT the geometric ladder of trace_probe_cone. The ladder's step grows with
// distance (step = aperture·dist, up to 0.5·dist for a close light), so whether a sample lands
// inside an occluder is a function of the probe→light distance — the shadow renders as
// concentric WAVES whose period grows with distance. Uniform steps sample every voxel along the
// ray exactly ~once at every distance, so occlusion is continuous in distance by construction.
// The cone LOD (diameter = 2·aperture·dist) is kept ONLY for the penumbra softness of the
// lookup; the step does not follow it. Opacity is corrected for the actual step length
// (1 − (1−a)^(step/voxel)) so a budget-floored longer step still accumulates the same total.
// Returns the same premultiplied (bleed.rgb, occlusion.a) contract as trace_probe_cone.
fn trace_shadow_cone(origin: vec3<f32>, dir: vec3<f32>, aperture: f32, reach: f32, maxSteps: i32, alphaCut: f32) -> vec4<f32> {
  var col = vec3<f32>(0.0);
  var alpha = 0.0;
  let voxelSize = uGridOrigin.w;
  let gridMin = uGridOrigin.xyz;
  let extent = vec3<f32>(uGridDims.xyz) * voxelSize;
  // ~1 voxel per step (the reference uses 0.9·VOXEL_SIZE), floored so maxSteps always covers
  // the full reach — a long reach trades sampling density, never silently truncates the shadow.
  let step = max(0.9 * voxelSize, reach / f32(maxSteps));
  let corr = step / voxelSize;
  var dist = voxelSize;
  for (var i = 0; i < maxSteps; i = i + 1) {
    if (alpha >= alphaCut || dist > reach) { break; }
    let wp = origin + dir * dist;
    let uvw = (wp - gridMin) / extent;
    if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) { break; }
    let diameter = max(voxelSize, 2.0 * aperture * dist);
    let lod = log2(diameter / voxelSize);
    let s = sample_radiance(uvw, lod, dir);
    let a = clamp(s.a, 0.0, 1.0);
    let ac = 1.0 - pow(1.0 - a, corr);
    // rgb is premultiplied by coverage — rescale it by the same correction ratio.
    let rc = select(s.rgb * (ac / max(a, 1e-4)), vec3<f32>(0.0), a <= 1e-4);
    col = col + (1.0 - alpha) * rc;
    alpha = alpha + (1.0 - alpha) * ac;
    dist = dist + step;
  }
  return vec4<f32>(col, alpha);
}

// Wang hash — the per-probe, per-frame random phase of the aimed systematic resampling. Anything
// LINEAR in globalSlot must not reach the light selection: slot is an atlas coord (y*gw + x), so a
// (slot mod lc)-style phase tiles the probe grid with period-lc bands the resolve renders as
// coherent moving stripes; the hash decorrelates probes AND frames into high-frequency noise the
// bilateral resolve + temporal history integrate away.
fn wang_hash(v: u32) -> u32 {
  var s = (v ^ 61u) ^ (v >> 16u);
  s = s * 9u;
  s = s ^ (s >> 4u);
  s = s * 0x27d4eb2du;
  s = s ^ (s >> 15u);
  return s;
}

// UNSHADOWED importance of emitter j (uLights index) seen from surface point P with normal N —
// the same atten/horizon/range-window math as the aimed trace loop, minus the shadow cone. It
// only steers the per-frame light SELECTION (the systematic resampling in the aimed loop); the
// traced contribution is divided by it, so any positive approximation keeps the estimator
// unbiased. Contract: returns 0 exactly when the trace loop would contribute nothing (the same
// culls, keeping the selection from wasting picks), > 0 otherwise.
fn emitter_importance(j: i32, P: vec3<f32>, N: vec3<f32>) -> f32 {
  let posR = uLights[2 * j];
  let colI = uLights[2 * j + 1];
  let toL = posR.xyz - P;
  let dc = length(toL);
  if (dc < 1e-3) { return 0.0; }
  let lr = max(posR.w, 1e-3);
  let horizon = clamp((dot(N, toL) / lr) * 0.5 + 0.5, 0.0, 1.0);
  if (horizon <= 0.0) { return 0.0; }
  let d = max(dc, lr);
  var atten = 1.0 / (1.0 + EMITTER_FALLOFF * (d * d) / (lr * lr));
  if (EMITTER_FALLOFF > 0.0) {
    let maxLumED = max(colI.r, max(colI.g, colI.b)) * abs(colI.w) * EMITTER_DIRECT;
    let r2 = lr * lr * max(maxLumED / 0.003 - 1.0, 0.0) / EMITTER_FALLOFF;
    let q = (d * d) / max(r2, 1e-6);
    let x = clamp(1.0 - q * q, 0.0, 1.0);
    atten = atten * x * x;
  }
  let lum = max(colI.r, max(colI.g, colI.b)) * abs(colI.w) * atten;
  if (lum * EMITTER_DIRECT < 0.003) { return 0.0; }
  return lum * horizon;
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
  let P = unproject(ndc, uInvViewProj);
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
  // ===== STAGE 3: history reprojection + validation, hoisted BEFORE the cone loop so the
  // golden-angle rotation below can be gated on whether THIS probe actually has usable history
  // (a probe whose history is persistently rejected — disocclusion, plane mismatch, off-screen —
  // must NOT rotate: it would trace a different cone subset every frame with zero accumulation
  // to average it back, i.e. steady-state flicker). Blend weight h = hyst · bilateral w; the
  // blend itself still happens after the loop, on the accumulated fresh SH.
  let hyst = uTemporalParams.x;
  var histTexel = vec2<i32>(0);
  var h = 0.0;
  if (hyst > 0.0) {
    // World anchor P → PREV clip. w <= 0 = behind last frame's camera → no history.
    let prevClip = uPrevViewProj * vec4<f32>(P, 1.0);
    if (prevClip.w > 1e-4) {
      let prevNdc = prevClip.xyz / prevClip.w;
      // EXACT inverse of the uv→ndc mapping above (note the Y flip): uv = (ndc.x·0.5 + 0.5,
      // 0.5 − ndc.y·0.5). Reverse-Z: ndc.z ∈ [0,1] on-frustum (near = 1, far = 0).
      let prevUv = vec2<f32>(prevNdc.x * 0.5 + 0.5, 0.5 - prevNdc.y * 0.5);
      if (all(prevUv >= vec2<f32>(0.0)) && all(prevUv < vec2<f32>(1.0))
          && prevNdc.z >= 0.0 && prevNdc.z <= 1.0) {
        // Prev pixel → prev UNIFORM tile texel (rows [0, gh) only — the uniform block is the
        // identity-mapped persistent backbone; adaptive history rows are never addressed because
        // adaptive slot order is not stable across frames).
        histTexel = clamp(
          vec2<i32>(prevUv * screenParams.xy / screenParams.z),
          vec2<i32>(0),
          vec2<i32>(gw - 1, gh - 1),
        );
        let hPos = textureLoad(histPos, histTexel, 0).xyz;
        let hNrmRaw = textureLoad(histNrm, histTexel, 0).xyz;
        // Validity: an invalid probe (and a freshly recreated zero-filled history texture) stored a
        // ZERO normal → |n|² ≈ 0 rejects it without a dedicated validity channel.
        if (dot(hNrmRaw, hNrmRaw) > 0.25) {
          // The SHARED refine/resolve bilateral (sp_plane_normal_weight) doubles as the history
          // validation — same semantics, one source of truth: (P, N) = the fresh probe, (Pp, Np) =
          // the history probe. Disocclusion (depth step / face flip) ⇒ w → 0 ⇒ fresh-only.
          let w = sp_plane_normal_weight(
            P, N, hPos, normalize(hNrmRaw),
            max(uTemporalParams.z, 1e-4), uTemporalParams.w,
          );
          h = hyst * w;
        }
      }
    }
  }

  // STAGE 3: per-frame golden-angle rotation of the whole Fibonacci set (frame index rides
  // uTemporalParams.y) so successive frames sample DIFFERENT directions and the temporal blend
  // integrates them back to an effective C × 1/(1-hysteresis) cone budget. GATED on this probe's
  // usable-history weight h (not just the global hysteresis): without history the rotation would
  // just make single-frame output flicker frame-to-frame, and the gate keeps hysteresis = 0
  // byte-identical to the pre-temporal build (the parity check).
  let phiOff = select(0.0, f32(i32(uTemporalParams.y)) * 2.39996323, h > 0.01);
  for (var i = 0; i < C; i = i + 1) {
    let cosT = 1.0 - (f32(i) + 0.5) / f32(C);   // [0,1) → upper hemisphere (cosT = component along N)
    let sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
    let phi = f32(i) * 2.39996323 + phiOff;     // golden angle (+ the per-frame rotation)
    let dir = nBasis * vec3<f32>(sinT * cos(phi), sinT * sin(phi), cosT); // local hemisphere → world

    let r = trace_probe_cone(origin, dir, aperture, reach, 64, 1.0);

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
  // AIMED emitter cones — traced once per PROBE (not per pixel): one narrow cone per emitter,
  // pointed straight at the light center, aperture = the light's angular size (radius / distance)
  // = the penumbra width. The analytic direct + bleed-cancel math is the classic per-pixel
  // formulation, with TWO deliberate changes for SH storage:
  //   - NO ndl factor in the stored value: the atlas holds RADIANCE SH and sh_avg_radiance applies
  //     the cosine at RESOLVE time — multiplying here would apply it twice (emitters would come out
  //     too dim at grazing angles). Only the ndl<=0 hemisphere REJECT survives (and the bleed term
  //     likewise drops its ndl).
  //   - weight = W_AIMED (a delta-direction projection constant — see its comment), NOT dw.
  // EMITTER_DIRECT is folded in HERE: the emitter light now rides the fill SH, which the cone pass
  // scales by GI_STRENGTH (default 1 → overall scale matches the old EMITTER_DIRECT-only path).
  // IMPORTANCE-DRIVEN SUBSAMPLE (unlimited lights at constant cost, MegaLights-style): when
  // lc > AIMED_PER_FRAME the probe picks "take" lights by SYSTEMATIC RESAMPLING over the
  // unshadowed-contribution weights (emitter_importance): the weight CDF is cut into take equal
  // strata, one pick each, shared random phase u. A light whose weight exceeds wSum/take is picked
  // EVERY frame deterministically — dominant lights never strobe, which is what makes the temporal
  // EMA converge (the old round-robin window alternated ceil(lc/take) fixed subsets, and an EMA of
  // an alternating sequence never converges — it oscillates at amplitude (1-h)/(1+h)*|A-B|,
  // the steady-state flicker). Only the dim tail is stochastic; each pick is scaled by
  // wSum/(take*wj) (m picks fold into one trace scaled by m), so the summed energy stays unbiased
  // in expectation and the Stage-3 blend integrates the residual (small) variance away.
  // With lc <= AIMED_PER_FRAME: every positive-weight light is traced exactly, scale 1.
  // CLUSTERED CULL: lc and the light indices come from THIS probe's cluster cell (uLightClusters —
  // CPU-binned by influence sphere), not the global list, so both the loop length and the
  // selection weights track the LOCAL light density. A probe outside the grid AABB gets lc = 0
  // (its lighting is fill-cones/sun only — same as before, the voxel field ends there anyway).
  let clusterDims = (uGridDims.xyz + vec3<i32>(CLUSTER_DIV - 1)) / vec3<i32>(CLUSTER_DIV);
  let clusterCell = vec3<i32>(floor((origin - uGridOrigin.xyz) / (uGridOrigin.w * f32(CLUSTER_DIV))));
  var lc = 0;
  var cbase = 0;
  if (all(clusterCell >= vec3<i32>(0)) && all(clusterCell < clusterDims)) {
    let ci = (clusterCell.z * clusterDims.y + clusterCell.y) * clusterDims.x + clusterCell.x;
    cbase = ci * (CLUSTER_CAP + 1);
    lc = min(i32(uLightClusters[cbase]), CLUSTER_CAP);
  }
  let take = min(lc, AIMED_PER_FRAME);
  // Selection weights over this probe's cluster list (0 = culled, never picked). lc is capped at
  // CLUSTER_CAP so the scratch array is fixed-size.
  var wArr: array<f32, CLUSTER_CAP>;
  var wSum = 0.0;
  for (var jj = 0; jj < lc; jj = jj + 1) {
    let wj = emitter_importance(i32(uLightClusters[cbase + 1 + jj]), P, N);
    wArr[jj] = wj;
    wSum = wSum + wj;
  }
  // Random stratum phase, decorrelated across probes (slot) AND frames (frame index) — see
  // wang_hash's comment for why nothing linear in the slot may reach the selection.
  let u = f32(wang_hash((u32(globalSlot) * 2654435761u) ^ u32(i32(uTemporalParams.y))) & 0xffffffu)
    / 16777216.0;
  var cum = 0.0;
  var k = 0;
  for (var jj = 0; jj < lc; jj = jj + 1) {
    let wj = wArr[jj];
    if (wj <= 0.0) { continue; }
    cum = cum + wj;
    // Systematic resampling: pick light jj once per stratum boundary (k + u)/take * wSum crossed
    // by the CDF at cum. m picks collapse into ONE trace scaled by m * wSum / (take * wj).
    var pickScale = 1.0;
    if (lc > take) {
      var m = 0.0;
      while (k < take && (f32(k) + u) * wSum <= cum * f32(take)) { m = m + 1.0; k = k + 1; }
      if (m <= 0.0) { continue; }
      pickScale = m * wSum / (f32(take) * wj);
    }
    let j = i32(uLightClusters[cbase + 1 + jj]);
    let posR = uLights[2 * j];        // .xyz center, .w radius
    let colI = uLights[2 * j + 1];    // .rgb color, .w intensity
    // Light geometry from the SURFACE point P, NOT the lifted origin: the lift (1.5·cellSize +
    // NORMAL_BIAS) exists only to start the MARCH clear of the surface voxel. Using it for the
    // direction/hemisphere test made a light hovering closer than the lift height point DOWNWARD
    // for the probes right under it → hemisphere-rejected → the light vanished at a distance
    // proportional to the voxel size.
    let toL = posR.xyz - P;
    let dc = length(toL);
    if (dc < 1e-3) { continue; }
    let dir = toL / dc;
    let lr = max(posR.w, 1e-3);
    // SPHERE-LIGHT HORIZON FADE replaces the binary hemisphere reject. A sphere half-sunk into the
    // surface has its CENTER at the horizon: dot(N, dir) flips sign probe-to-probe → neighboring
    // probes alternate between the FULL (atten ≈ 1, huge) contribution and ZERO, and the resolve
    // renders that lattice as noise. The visible-cap fraction (center height above the tangent
    // plane / diameter, remapped to [0,1]) fades the light smoothly through the horizon instead.
    let horizon = clamp((dot(N, toL) / lr) * 0.5 + 0.5, 0.0, 1.0);
    if (horizon <= 0.0) { continue; }   // fully below the surface plane
    // Probe at/inside the light sphere: clamp the shading distance to the sphere surface so atten
    // and the aperture stay finite and continuous (dc still drives the shadow reach below).
    let d = max(dc, lr);
    // Distance falloff: atten = 1 at the light center, ~1/d² far. EMITTER_FALLOFF = coefficient
    // (0 = none → flat sun-like emitter, never culled below; 1 = standard); lr = emitter radius.
    var atten = 1.0 / (1.0 + EMITTER_FALLOFF * (d * d) / (lr * lr));
    // RANGE WINDOW (the UE4/Frostbite windowed falloff, (1 − (d/R)⁴)²): scales the light SMOOTHLY
    // to exactly zero at R = the CPU binning radius (both derive R from the same 0.003 cull:
    // R² = lr²·(maxLum·EMITTER_DIRECT/0.003 − 1)/EMITTER_FALLOFF). Without it the light ends with
    // a 0.003-high STEP at the last binned cluster cell — visible as cell-boundary "sectors" on
    // the floor. EMITTER_FALLOFF = 0 ⇒ infinite range ⇒ no window (and the CPU bins to all cells).
    if (EMITTER_FALLOFF > 0.0) {
      let maxLumED = max(colI.r, max(colI.g, colI.b)) * abs(colI.w) * EMITTER_DIRECT;
      let r2 = lr * lr * max(maxLumED / 0.003 - 1.0, 0.0) / EMITTER_FALLOFF;
      let q = (d * d) / max(r2, 1e-6);
      let x = clamp(1.0 - q * q, 0.0, 1.0);
      atten = atten * x * x;
    }
    let Lj = colI.rgb * abs(colI.w) * atten;
    // LIGHT CULL: the shadow march is the dominant cost — skip it when the final contribution is
    // negligible (same 0.003 gate as the old per-pixel loop, sans ndl: a probe serves pixels of
    // many normals, so the cull must be normal-agnostic → strictly more conservative).
    if (max(Lj.r, max(Lj.g, Lj.b)) * EMITTER_DIRECT < 0.003) { continue; }
    let ap = clamp(posR.w / d, 0.02, 0.5);   // angular size of the light = penumbra width
    // SHADOW REACH stops at the light's SURFACE minus one voxel — marching to the CENTER (old
    // reach = d) always ended inside the emitter's own voxelized cells, self-occluding every
    // close light (occ→1 while trilinear/LOD dilutes the bleed below the shadow). NEAR-FIELD
    // GATE: within ~3 voxels the field cannot resolve an occluder at all (the march starts at
    // dist = voxelSize and the surface+light share cells) — skip the trace, take occ = 0.
    let shadowReach = dc - lr - uGridOrigin.w;
    var r = vec4<f32>(0.0);
    if (shadowReach > uGridOrigin.w * 3.0) {
      r = trace_shadow_cone(origin, dir, ap, shadowReach, max(1, AIMED_STEPS), AIMED_ALPHA_CUT);
    }
    // ANALYTIC DIRECT + bleed-cancel (the "white shadow" fix): shadow = what the cone's opacity
    // removes; bleed = the radiance the cone actually gathered (a BRIGHT occluder => big bleed =>
    // its false shadow is cancelled; a DARK occluder => ~0 bleed => the shadow survives). Clamped
    // so it can only darken, never exceed Lj.
    let occ = clamp(r.a, 0.0, 1.0);
    let shadow = Lj * occ;
    let bleed = r.rgb;
    let contrib = max(vec3<f32>(0.0), Lj - max(vec3<f32>(0.0), shadow - bleed))
      * (EMITTER_DIRECT * pickScale * horizon);
    // Project into the SAME SH-L1 accumulators as the fill cones, at the emitter direction.
    let Y00 = 0.282095;
    let Y1m1 = 0.488603 * dir.y;
    let Y10 = 0.488603 * dir.z;
    let Y11 = 0.488603 * dir.x;
    let shb = vec4<f32>(Y00, Y1m1, Y10, Y11) * W_AIMED;
    cR = cR + contrib.r * shb;
    cG = cG + contrib.g * shb;
    cB = cB + contrib.b * shb;
  }

  // ===== STAGE 3: temporal blend — h was computed (reproject + validate) BEFORE the cone loop.
  // h = 0 leaves the fresh SH untouched (the exact pre-temporal path). Every failure mode (behind
  // the prev camera, off-screen, invalid/zeroed history, plane/normal disocclusion) landed h = 0.
  // Raw-SH lerp is valid (SH is linear in its coefficients; both sets are the same
  // solid-angle-weighted RADIANCE encoding).
  if (h > 0.0) {
    let hR = textureLoad(histShR, histTexel, 0);
    let hG = textureLoad(histShG, histTexel, 0);
    let hB = textureLoad(histShB, histTexel, 0);
    // CHANGE DETECTION (the RTXGI/DDGI adaptive-hysteresis scheme): a REAL lighting change (a
    // light moved/appeared) must not be dragged out over the 1/(1-h) history lag — that lag is
    // the ghosting a high global hysteresis buys. Relative DC-luminance difference fresh vs
    // history: > 25% shaves the history weight, > 80% drops it entirely (instant response).
    // Sampling noise stays below the 25% gate: the aimed lights are importance-sampled (dominant
    // lights are traced every frame) and the fill-cone rotation only redistributes directions.
    let freshLum = dot(vec3<f32>(cR.x, cG.x, cB.x), vec3<f32>(0.2126, 0.7152, 0.0722));
    let histLum = dot(vec3<f32>(hR.x, hG.x, hB.x), vec3<f32>(0.2126, 0.7152, 0.0722));
    let change = abs(freshLum - histLum) / max(max(abs(freshLum), abs(histLum)), 1e-3);
    if (change > 0.25) { h = max(0.0, h - 0.15); }
    if (change > 0.8) { h = 0.0; }
    cR = mix(cR, hR, h);
    cG = mix(cG, hG, h);
    cB = mix(cB, hB, h);
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
