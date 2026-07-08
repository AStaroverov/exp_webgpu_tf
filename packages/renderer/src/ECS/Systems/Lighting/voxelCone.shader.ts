import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";
import { VoxelBakedConfig } from "./voxelConfig.ts";
import { SCREEN_PROBE_K } from "./voxelResources.ts";
import { probeWeightWGSL } from "./voxelProbeShared.wgsl.ts";

// VCT Layer 3 — the full DIFFUSE HEMISPHERE cone gather. A fullscreen pass over the G-buffer:
//   1. Reconstruct the per-pixel world position P (from the reverse-Z depth + invViewProj)
//      and world normal N (from the normal G-buffer; n.rgb*2-1, n.a<0.5 = no surface).
//   2. Trace N cones over the hemisphere around the normal through the voxelRadiance mip
//      pyramid, on a procedural Fibonacci (golden-angle) hemisphere with cosine weights.
//      Each cone's diameter grows with distance (diameter = 2*aperture*dist); each step
//      samples the pyramid at LOD = log2(diameter / voxelSize) — a wider cone reads a
//      coarser mip, so one filtered fetch integrates the whole cone cross-section.
//      Front-to-back "over" compositing accumulates radiance + opacity per cone, then the
//      cones are cosine-weight-averaged → the cosine-weighted AVERAGE incoming radiance.
//   3. Write the gathered radiance (scaled by giStrength) to an HDR texture.
//
// ANGULAR RESOLUTION = SHADOW SHARPNESS. The cone count + aperture together set how finely
// the hemisphere is sampled, and per-direction occupancy (cone.a) IS the shadow term: MORE
// cones + a NARROWER aperture = sharper umbra/penumbra. But a too-narrow aperture with
// too-few cones leaves angular GAPS between cones (banding) — raise the cone count whenever
// you narrow the aperture.
//
// PERF: this pass runs at a DOWNSCALED resolution (half by default → ¼ the pixels → ~4× less
// cone work; quarter → 1/16). The downscale factor is chosen on the CPU via the cone output size;
// the shader is resolution-agnostic (maps via texCoord). The composite normal-aware-upsamples
// this output back to full res — indirect light is low-frequency, so that is fine.
//
// IMPORTANCE-SAMPLED HEMISPHERE GATHER. Wide Fibonacci fill cones at a 60° aperture read coarse
// mips, so they smear a small bright emitter across the whole cone → too dim + no directional
// shadow if used alone. So we ALSO aim one narrow cone straight at each emitter (full reach to the
// source, no fade) for a bright, sharply-shadowed contribution, folded into the SAME cosine-
// weighted hemisphere integral (acc/occAcc/wsum). The emitters are AUTO-DISCOVERED from the
// LightEmitter component every frame (no manual light list) — every emitter is treated the same.
//
// This is what produces real color bleeding. The earlier single-cone-along-the-normal form
// was the Layer-2 INTERMEDIATE (a bent-normal / AO preview); the hemisphere gather here is
// directly comparable to the brute-force gi reference, which is also a cosine-weighted
// hemisphere average.
//
// Fullscreen pass with `unproject` + reverse-Z NDC reconstruction. textureSampleLevel (explicit
// LOD) is used in the trace loop — legal in non-uniform control flow (unlike textureSample).

export function createConeShaderMeta(cfg: VoxelBakedConfig) {
  return new ShaderMeta(
  {
    // .x = screen width (px), .y = screen height (px), .z = anisoMode (0 = isotropic pyramid,
    // 1 = anisotropic directional volumes — the far-field anti-leak), .w = active light count (0..8).
    params2: new VariableMeta("uParams2", VariableKind.Uniform, `vec4<f32>`),
    // Screen-probe resolve params (all LIVE per-frame uniforms — GUI-tunable with no rebuild):
    // .x = SCREEN_PROBE_TILE (full-res px / probe), .y = normal-weight power (SP_NORMAL_POW),
    // .z = plane-threshold scale (SP_PLANE_K, × local probe spacing), .w = resolveRadius (the smooth
    // screen kernel's support in TILES — bigger = smoother/wider fill, smaller = more local detail).
    params3: new VariableMeta("uParams3", VariableKind.Uniform, `vec4<f32>`),
    // Emitters to importance-sample: .xyz = world CENTER, .w = radius (penumbra source). These
    // are AUTO-DISCOVERED from the LightEmitter component (every emitter, no manual list); only
    // the first i32(uParams2.w) entries are live.
    lights: new VariableMeta("lights", VariableKind.Uniform, `array<vec4<f32>, 8>`),
    // Parallel to lights[]: .rgb = emitter color, .w = intensity → Lj = rgb·|w| is the emitter's
    // true radiance. Used to compute analytic direct light (so a blocked cone DARKENS instead of
    // picking up the occluder's own emission = the "white shadow" bug), with a bleed term so a
    // BRIGHT occluder cancels its own false shadow.
    lightColor: new VariableMeta("lightColor", VariableKind.Uniform, `array<vec4<f32>, 8>`),
    // inverse(viewProjMatrix) (reverse-Z), column-major, for world-position reconstruction.
    invViewProj: new VariableMeta("uInvViewProj", VariableKind.Uniform, `mat4x4<f32>`),
    // .xyz = world min corner, .w = cellSize.
    gridOrigin: new VariableMeta("uGridOrigin", VariableKind.Uniform, `vec4<f32>`),
    // .xyz = voxel counts per axis.
    gridDims: new VariableMeta("uGridDims", VariableKind.Uniform, `vec4<i32>`),
    // G-buffer reverse-Z depth (texture_depth_2d) + world normal (rgba16float, packed *0.5+0.5).
    depthTex: new VariableMeta("depthTex", VariableKind.Texture, `texture_depth_2d`, {
      textureSampleType: "depth",
    }),
    normalTex: new VariableMeta("normalTex", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "float",
    }),
    // The voxelRadiance mip pyramid (ALL mips) — the cone reads it at the per-step LOD (for the
    // aimed emitter cones + the short AO cones).
    voxelRadiance: new VariableMeta("voxelRadiance", VariableKind.Texture, `texture_3d<f32>`, {
      viewDimension: "3d",
      textureSampleType: "float",
    }),
    // The 6 ANISOTROPIC directional radiance volumes (−X,+X,−Y,+Y,−Z,+Z), each an ALL-mips
    // sampled view of the half-res directional pyramid built by voxelAnisoBase/voxelAnisoVolume.
    // sample_aniso picks the 3 facing the cone dir and blends by dir² → direction-correct occlusion
    // for the far-field (coarse-LOD) samples. The near field still reads voxelRadiance mip 0 (crisp).
    anisoNegX: new VariableMeta("anisoNegX", VariableKind.Texture, `texture_3d<f32>`, {
      viewDimension: "3d",
      textureSampleType: "float",
    }),
    anisoPosX: new VariableMeta("anisoPosX", VariableKind.Texture, `texture_3d<f32>`, {
      viewDimension: "3d",
      textureSampleType: "float",
    }),
    anisoNegY: new VariableMeta("anisoNegY", VariableKind.Texture, `texture_3d<f32>`, {
      viewDimension: "3d",
      textureSampleType: "float",
    }),
    anisoPosY: new VariableMeta("anisoPosY", VariableKind.Texture, `texture_3d<f32>`, {
      viewDimension: "3d",
      textureSampleType: "float",
    }),
    anisoNegZ: new VariableMeta("anisoNegZ", VariableKind.Texture, `texture_3d<f32>`, {
      viewDimension: "3d",
      textureSampleType: "float",
    }),
    anisoPosZ: new VariableMeta("anisoPosZ", VariableKind.Texture, `texture_3d<f32>`, {
      viewDimension: "3d",
      textureSampleType: "float",
    }),
    // SCREEN-SPACE probe SH-L1 textures (the low-frequency diffuse fill/bounce source; .xyzw = the
    // 4 SH coeffs). 2D (one texel per screen probe), point-loaded (textureLoad) in
    // resolve_screen_probes — no sampler.
    screenShR: new VariableMeta("screenShR", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "float",
    }),
    screenShG: new VariableMeta("screenShG", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "float",
    }),
    screenShB: new VariableMeta("screenShB", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "float",
    }),
    // Per-probe geometry: .xy = representative full-res pixel, .z = validity, .w = the probe FOOTPRINT
    // (cell size in full-res px) → the resolve area-weights each probe by footprint² (density-invariant
    // average). rgba32float → declared "unfilterable-float" (point-loaded); the resolve reconstructs
    // each probe's P + N from the G-buffer at .xy. ALWAYS bound (pruning-safe).
    screenProbePix: new VariableMeta("screenProbePix", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "unfilterable-float",
    }),
    // Per-probe world anchor P (.xyz) + world normal N (.xyz), written by the gather. The resolve
    // point-loads these instead of reconstructing each probe's P/N from the full-res G-buffer per tap
    // (P1). Both unfilterable-float (point sampling). ALWAYS bound (pruning-safe).
    screenProbePos: new VariableMeta("screenProbePos", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "unfilterable-float",
    }),
    screenProbeNrm: new VariableMeta("screenProbeNrm", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "unfilterable-float",
    }),
    // ADAPTIVE screen-probe indirection — group 1 (StorageRead, read-only storage in the fragment
    // stage, allowed in core WebGPU). tileHeader[T] = count of adaptive probes parented to coarse
    // tile T; tileIndices[T*K + j] = the global atlas slot of the tile's j-th adaptive probe. When
    // lightThresh is high (no adaptive probes) tileHeader is all-zero (cleared, never written), so
    // the adaptive-tap loop below is skipped and the resolve is identical to the flat-atlas build.
    tileHeader: new VariableMeta("uTileHeader", VariableKind.StorageRead, `array<u32>`, {
      visibility: GPUShaderStage.FRAGMENT,
    }),
    tileIndices: new VariableMeta("uTileIndices", VariableKind.StorageRead, `array<u32>`, {
      visibility: GPUShaderStage.FRAGMENT,
    }),
    // Filtering sampler for textureSampleLevel over the voxelRadiance pyramid + the aniso volumes.
    voxelSampler: new VariableMeta("voxelSampler", VariableKind.Sampler, `sampler`),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
// BAKED tuning consts (interpolated from VoxelBakedConfig at shader-build time). See voxelConfig.ts.
const NORMAL_BIAS: f32 = ${cfg.normalBias};
const APERTURE: f32 = ${cfg.aperture};
const GI_STRENGTH: f32 = ${cfg.giStrength};
const EMITTER_FALLOFF: f32 = ${cfg.emitterFalloff};
const EMITTER_DIRECT: f32 = ${cfg.emitterDirect};
const AIMED_STEPS: i32 = ${cfg.aimedSteps};
const AIMED_ALPHA_CUT: f32 = ${cfg.aimedAlphaCut};
const AO_CONE_COUNT: i32 = ${cfg.aoConeCount};
const AO_REACH: f32 = ${cfg.aoReach};
const AO_STEPS: i32 = ${cfg.aoSteps};
// Max adaptive probes per coarse tile = the tileIndices stride (see voxelResources.SCREEN_PROBE_K).
const SP_K: u32 = ${SCREEN_PROBE_K}u;

${probeWeightWGSL}

const POSITION = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
  vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);
const TEX_COORDS = array<vec2f, 6>(
  vec2f(0.0, 1.0), vec2f(1.0, 1.0), vec2f(1.0, 0.0),
  vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(0.0, 0.0)
);

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) texCoord: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var out: VertexOutput;
  out.position = vec4f(POSITION[vertexIndex], 0.0, 1.0);
  out.texCoord = TEX_COORDS[vertexIndex];
  return out;
}

// Unproject an NDC point (z reverse-Z) to world space.
fn unproject(ndc: vec3<f32>) -> vec3<f32> {
  let w = uInvViewProj * vec4<f32>(ndc, 1.0);
  return w.xyz / w.w;
}

// Orthonormal basis with column 2 = n (so basis * (x,y,z) = x*t + y*b + z*n).
fn build_basis(n: vec3<f32>) -> mat3x3<f32> {
  let a = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.9);
  let t = normalize(cross(a, n));
  let b = cross(n, t);
  return mat3x3<f32>(t, b, n);
}

// Sample the 6 ANISOTROPIC directional volumes for a cone traveling along unit dir, at aniso LOD.
// Per axis pick the volume whose pre-integration faces the ray (sign of dir): the negX volume is
// accumulated front-to-back for a ray heading toward −X (built from the +X side), so dir.x<0 reads
// negX, dir.x>0 reads posX (same for Y/Z). Blend the 3 by dir² (Σ dir² = 1 for a unit dir) → the
// direction-correct occluded radiance. Each branch is one fetch (3 total), legal in non-uniform
// control flow because textureSampleLevel takes an explicit LOD.
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

// The radiance fetch inside the cone march. uParams2.z toggles the model:
//   iso  (0): the plain isotropic voxelRadiance pyramid (one direction-agnostic average).
//   aniso(1): near field (lod≲1) reads the crisp iso mip 0; the far field reads the directional
//             volumes at (lod-1) — aniso level 0 == iso mip 1 resolution — blended over lod∈[0,1]
//             so there is no seam where the two representations meet.
// Both volumes store PREMULTIPLIED rgb (radiance·coverage) + coverage in .a, so the caller's
// front-to-back "over" operator is unchanged regardless of which model is active.
fn sample_radiance(uvw: vec3<f32>, lod: f32, dir: vec3<f32>) -> vec4<f32> {
  if (uParams2.z < 0.5) {
    return textureSampleLevel(voxelRadiance, voxelSampler, uvw, lod);
  }
  let iso0 = textureSampleLevel(voxelRadiance, voxelSampler, uvw, min(lod, 1.0));
  let aniso = sample_aniso(uvw, max(0.0, lod - 1.0), dir);
  return mix(iso0, aniso, clamp(lod, 0.0, 1.0));
}

// One cone marched along dir from origin: diameter grows with distance, each step samples
// voxelRadiance at LOD=log2(diameter/voxelSize) and composites front-to-back ("over").
// reach = how far this cone marches (world units). The step is floored at reach/maxSteps so the
// cone ALWAYS spans its full reach within the step budget — without this, a narrow cone's
// tiny near-field steps burn the budget and it dies ~16 units out (the "light stops at cone
// reach" problem). fadeFrac>0 tapers the gathered radiance over the last fadeFrac of reach
// (hides the artificial cutoff of the fill cones); pass 0 for aimed cones, which end at the
// real light, so they must NOT be tapered. Occlusion (alpha) is never tapered → shadows stay.
// maxSteps = march budget (aimed cones want it high for sharp shadows; the low-frequency fill
// hemisphere is fine with ~half — its bounce is smooth, so fewer steps barely change it but
// halve the dominant cone cost). alphaCut = early-out opacity: a fill cone that is ~opaque adds
// almost nothing more, so cutting at <1 saves the tail steps; aimed cones pass 1.0 (no early cut).
fn trace_cone(origin: vec3<f32>, dir: vec3<f32>, aperture: f32, reach: f32, fadeFrac: f32, startJ: f32, maxSteps: i32, alphaCut: f32) -> vec4<f32> {
  var col = vec3<f32>(0.0);
  var alpha = 0.0;
  let voxelSize = uGridOrigin.w;
  let gridMin = uGridOrigin.xyz;
  let extent = vec3<f32>(uGridDims.xyz) * voxelSize;
  let stepFloor = reach / f32(maxSteps);
  // Per-pixel SCALE of the start distance (startJ in [0,1)). Because the march is ~geometric
  // (step grows with distance), scaling dist0 shifts the sampling "shells" by a per-pixel
  // factor at EVERY distance → the concentric "tree-ring" banding around bright sources
  // dithers across pixels and is blurred away by the half-res upsample. Lets few cones look
  // smooth without going to 48.
  var dist = voxelSize * (1.0 + startJ);
  for (var i = 0; i < maxSteps; i = i + 1) {
    if (alpha >= alphaCut || dist > reach) { break; }
    let diameter = max(voxelSize, 2.0 * aperture * dist);
    let lod = log2(diameter / voxelSize);
    let wp = origin + dir * dist;
    let uvw = (wp - gridMin) / extent;
    if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) { break; }
    let s = sample_radiance(uvw, lod, dir);
    var window = 1.0;
    if (fadeFrac > 0.0) {
      window = clamp((reach - dist) / (reach * fadeFrac), 0.0, 1.0);
    }
    col = col + (1.0 - alpha) * s.rgb * window;
    alpha = alpha + (1.0 - alpha) * s.a;
    dist = dist + max(diameter * 0.5, stepFloor);
  }
  return vec4<f32>(col, alpha);
}

// Reconstruct the cosine-weighted AVERAGE radiance from an SH-L1 channel (.xyzw = L00,L1m1,L10,L11)
// for surface normal N. The SH-L1 diffuse IRRADIANCE is E = 0.886227*L00 + 1.023328*(L1·N); we
// divide by PI to get the average incoming radiance (for uniform radiance L, E/PI = L), so this
// matches the scale of the OLD per-cone fill average it replaces. Clamped against SH ringing.
fn sh_avg_radiance(c: vec4<f32>, N: vec3<f32>) -> f32 {
  let E = 0.886227 * c.x + 1.023328 * (c.y * N.y + c.z * N.z + c.w * N.x);
  return max(0.0, E * 0.31830989); // * (1/PI)
}

// ---- SCREEN-PROBE FILL (the sole diffuse fill/bounce source; see the (b) fill block). ----
// UNIFIED RESOLVE. Every probe — uniform AND adaptive — flows through ONE identical path
// (accum_probe): there is NO uniform-vs-adaptive branch anywhere in the tap weighting or source.
// The old split (uniform = blurred SH + bilinear cage weight, adaptive = raw SH + a peaky gaussian)
// made dense/adaptive regions snap to their raw voxel-discrete SH and reveal the voxel grid while
// uniform-only stayed smooth. Now BOTH read the RAW SH atlas (shR/shG/shB) and BOTH use the SAME
// smooth compact screen-distance kernel, so uniform vs adaptive is invisible in the output and
// adding adaptive density only sharpens gradients — it never adds blockiness.

// Load a probe's world position from its atlas texel (stored by the gather). .w = ok (validity).
fn sp_load_probe_P(atexel: vec2<i32>) -> vec4<f32> {
  let pixMeta = textureLoad(screenProbePix, atexel, 0);   // .z valid
  if (pixMeta.z < 0.5) { return vec4<f32>(0.0); }
  return vec4<f32>(textureLoad(screenProbePos, atexel, 0).xyz, 1.0);
}

// The ONE per-probe tap (used identically for uniform and adaptive slots). Loads the probe's repr
// pixel, reconstructs its Pp/Np from the G-buffer, weights it by wSpatial (a SMOOTH compact kernel on
// SCREEN-pixel distance normalised by the resolve radius) and adds:
//   - rawSH * wSpatial into the LOOSE sum (the disocclusion backstop — no bilateral gate), and
//   - rawSH * (wSpatial * plane-normal weight) into the STRICT sum (the bilateral-gated average).
// wSpatial = max(0, 1 - d*d), d = |reprPixel - full| / (tile * resolveRadius): compact (0 past the
// support), smooth, and IDENTICAL for uniform and adaptive — that identity is the whole point.
fn accum_probe(
  atexel: vec2<i32>, P: vec3<f32>, N: vec3<f32>, full: vec2<f32>,
  tile: f32, resolveRadius: f32, planeThresh: f32, normalPow: f32,
  sumR: ptr<function, vec4<f32>>, sumG: ptr<function, vec4<f32>>, sumB: ptr<function, vec4<f32>>,
  wsum: ptr<function, f32>,
  looseR: ptr<function, vec4<f32>>, looseG: ptr<function, vec4<f32>>, looseB: ptr<function, vec4<f32>>,
  lsum: ptr<function, f32>,
) {
  let pixMeta = textureLoad(screenProbePix, atexel, 0);   // .xy pixel, .z valid, .w footprint (px)
  if (pixMeta.z < 0.5) { return; }
  let pc = pixMeta.xy;                                     // the probe's representative screen pixel
  // DENSITY COMPENSATION: area-weight the probe by the screen area it represents (its footprint² in
  // full-res px), so N fine adaptive probes collectively weigh the same as the 1 coarse uniform probe
  // they subdivide → the average is density-invariant (dense/adaptive regions match uniform ones, and
  // an adaptive spawn/despawn no longer shifts brightness). Folded into wSpatial ONCE, so BOTH the
  // loose (wSpatial) and strict (wSpatial × plane×normal) accumulators inherit it — the flow stays
  // UNIFIED (one accum_probe, no uniform-vs-adaptive branch; the footprint just rides in pixMeta.w).
  // NOTE: a subdivided tile slightly double-counts — the uniform probe still votes its tile² alongside
  // its children — a mild bias toward the coarse value that actually AIDS stability; a proper Voronoi
  // split is deferred.
  let areaW = pixMeta.w * pixMeta.w;
  if (areaW <= 0.0) { return; }
  // SMOOTH compact screen-distance kernel (normalised so support == resolveRadius tiles).
  let d = length(pc - full) / (tile * resolveRadius);
  let wSpatial = max(0.0, 1.0 - d * d) * areaW;
  if (wSpatial <= 0.0) { return; }
  // P1: the probe's world anchor + normal are read straight from the atlas (stored by the gather),
  // NOT reconstructed from the full-res G-buffer per tap. Np is stored normalized (half-float error is
  // negligible for the similarity weight).
  let Pp = textureLoad(screenProbePos, atexel, 0).xyz;
  let Np = textureLoad(screenProbeNrm, atexel, 0).xyz;
  let sR = textureLoad(screenShR, atexel, 0);
  let sG = textureLoad(screenShG, atexel, 0);
  let sB = textureLoad(screenShB, atexel, 0);
  // LOOSE (spatial-only) backstop — accumulated for every in-support probe, no bilateral gate.
  *looseR = *looseR + wSpatial * sR; *looseG = *looseG + wSpatial * sG; *looseB = *looseB + wSpatial * sB;
  *lsum = *lsum + wSpatial;
  // STRICT: multiply by the SHARED plane×normal weight (byte-identical to the refine placement test).
  let w = wSpatial * sp_plane_normal_weight(P, N, Pp, Np, planeThresh, normalPow);
  if (w <= 0.0) { return; }
  *sumR = *sumR + w * sR; *sumG = *sumG + w * sG; *sumB = *sumB + w * sB;
  *wsum = *wsum + w;
}

// Resolve the screen-probe fill at pixel 'full' (P, N = its world position + normal). Gathers a SCREEN
// NEIGHBOURHOOD of probes over a small window of uniform grid cells (radius rc around the pixel's grid
// coord), and for each in-bounds cell taps (a) that cell's UNIFORM probe and (b) that tile's ADAPTIVE
// probes (tileHeader up to SP_K, via tileIndices) — all through the SAME accum_probe. The smooth
// kernel makes the fill a spatially-continuous field with no per-probe snapping; a flat tile has
// tileHeader == 0 so its adaptive inner loop is empty (uniform-only parity). resolveRadius is a LIVE
// uniform (uParams3.w, GUI-tunable): bigger = smoother/wider support (also helps a distant object seen
// by few probes), smaller = more local detail.
// WORST-CASE TAPS: (2*rc+1)^2 uniform cells, each 1 uniform + up to SP_K adaptive probes →
// (2*rc+1)^2 * (1 + SP_K). Default rc=2, SP_K=8 → 25*9 = 225; typical far fewer (most tiles have
// zero/few adaptive probes and cells past the kernel support weight to 0).
fn resolve_screen_probes(P: vec3<f32>, N: vec3<f32>, full: vec2<i32>) -> vec3<f32> {
  let tile = uParams3.x;
  let normalPow = uParams3.y;
  let planeK = uParams3.z;
  let resolveRadius = max(0.25, uParams3.w);   // kernel support in TILES (guarded off zero)
  let gw = i32(ceil(uParams2.x / tile));
  let gh = i32(ceil(uParams2.y / tile));
  let cellSize = uGridOrigin.w;

  // The uniform grid cell containing this pixel (gf = full/tile) + the window half-extent in cells.
  let gf = vec2<f32>(full) / tile;
  let gc = vec2<i32>(floor(gf));
  let rc = clamp(i32(ceil(resolveRadius)), 1, 4);

  // Local world spacing between adjacent UNIFORM probes = the depth-adaptive plane tolerance (from
  // reconstructed Pp, so distant flat surfaces are NOT over-rejected by a fixed world-unit threshold).
  // Uniform slot = cy*gw+cx → atlas texel (cx,cy) (identity), so the cell coord IS the atlas texel.
  // ROBUST DERIVATION (silhouette rejection): take the MIN spacing over the (up to 4) CARDINAL
  // neighbours of the center probe. On a silhouette a single neighbour lands on the background across
  // the depth discontinuity → its spacing balloons and, if used alone, inflates planeThresh so the
  // background probe is NOT rejected (bleed). The discontinuity inflates only the crossing neighbour(s);
  // the MIN picks a same-surface neighbour → a tight planeThresh that rejects the background probe. On a
  // distant flat surface all four spacings are similarly large → min ≈ correct (no over-rejection).
  var spacingMin = cellSize * 4.0;                   // fallback if no valid cardinal neighbour
  let gcc = clamp(gc, vec2<i32>(0), vec2<i32>(gw - 1, gh - 1));
  let cP = sp_load_probe_P(gcc);
  if (cP.w > 0.5) {
    var best = 1e30;
    var found = false;
    if (gcc.x + 1 < gw) {
      let nP = sp_load_probe_P(vec2<i32>(gcc.x + 1, gcc.y));
      if (nP.w > 0.5) { best = min(best, length(nP.xyz - cP.xyz)); found = true; }
    }
    if (gcc.x - 1 >= 0) {
      let nP = sp_load_probe_P(vec2<i32>(gcc.x - 1, gcc.y));
      if (nP.w > 0.5) { best = min(best, length(nP.xyz - cP.xyz)); found = true; }
    }
    if (gcc.y + 1 < gh) {
      let nP = sp_load_probe_P(vec2<i32>(gcc.x, gcc.y + 1));
      if (nP.w > 0.5) { best = min(best, length(nP.xyz - cP.xyz)); found = true; }
    }
    if (gcc.y - 1 >= 0) {
      let nP = sp_load_probe_P(vec2<i32>(gcc.x, gcc.y - 1));
      if (nP.w > 0.5) { best = min(best, length(nP.xyz - cP.xyz)); found = true; }
    }
    if (found) { spacingMin = best; }
  }
  let planeThresh = max(cellSize, spacingMin) * planeK;

  // Strict (bilateral) + loose (spatial-only) sums, accumulated over the whole neighbourhood in one
  // pass. The loose sum is the disocclusion backstop: if every strict weight rejects, fall back to it.
  var sumR = vec4<f32>(0.0); var sumG = vec4<f32>(0.0); var sumB = vec4<f32>(0.0);
  var wsum = 0.0;
  var looseR = vec4<f32>(0.0); var looseG = vec4<f32>(0.0); var looseB = vec4<f32>(0.0);
  var lsum = 0.0;
  let fullF = vec2<f32>(full);
  for (var dy = -rc; dy <= rc; dy = dy + 1) {
    for (var dx = -rc; dx <= rc; dx = dx + 1) {
      let cx = gc.x + dx;
      let cy = gc.y + dy;
      if (cx < 0 || cy < 0 || cx >= gw || cy >= gh) { continue; }
      // (a) UNIFORM probe of this cell — atlas texel == cell coord (identity mapping).
      accum_probe(vec2<i32>(cx, cy), P, N, fullF, tile, resolveRadius, planeThresh, normalPow,
        &sumR, &sumG, &sumB, &wsum, &looseR, &looseG, &looseB, &lsum);
      // (b) ADAPTIVE probes parented to this tile (empty when tileHeader == 0 → uniform-only parity).
      let T = u32(cy * gw + cx);
      let cnt = min(uTileHeader[T], SP_K);
      for (var j = 0u; j < cnt; j = j + 1u) {
        let aslot = uTileIndices[T * SP_K + j];
        let atexel = vec2<i32>(i32(aslot) % gw, i32(aslot) / gw);
        accum_probe(atexel, P, N, fullF, tile, resolveRadius, planeThresh, normalPow,
          &sumR, &sumG, &sumB, &wsum, &looseR, &looseG, &looseB, &lsum);
      }
    }
  }

  if (wsum > 1e-4) {
    let inv = 1.0 / wsum;
    return vec3<f32>(
      sh_avg_radiance(sumR * inv, N),
      sh_avg_radiance(sumG * inv, N),
      sh_avg_radiance(sumB * inv, N));
  }
  if (lsum > 1e-4) {
    // Strict gate rejected all → loose spatial blend over the in-support probes (softens edges; may
    // bleed slightly at silhouettes, but never black — no world volume to fall back to).
    let inv = 1.0 / lsum;
    return vec3<f32>(
      sh_avg_radiance(looseR * inv, N),
      sh_avg_radiance(looseG * inv, N),
      sh_avg_radiance(looseB * inv, N));
  }
  return vec3<f32>(0.0);  // no valid probe anywhere in the window → no fill
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  // This pass renders at a downscaled res (half or quarter — set on the CPU by the cone output
  // size). Map this cone pixel to a representative full-res G-buffer texel via the normalized
  // texCoord, which spans [0,1] across the target at ANY resolution → no scale uniform needed.
  // uParams2.xy carries the FULL canvas dims. (half below is still the per-pixel jitter seed.)
  let half = vec2<i32>(floor(input.position.xy));
  let full = min(vec2<i32>(input.texCoord * uParams2.xy), vec2<i32>(uParams2.xy) - vec2<i32>(1));

  // World normal from the G-buffer; a<0.5 = no surface at this pixel.
  let n = textureLoad(normalTex, full, 0);
  if (n.a < 0.5) {
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }
  let N = normalize(n.rgb * 2.0 - 1.0);

  // Reconstruct world position from reverse-Z depth.
  let depth = textureLoad(depthTex, full, 0);
  let uv = (vec2<f32>(full) + vec2<f32>(0.5)) / uParams2.xy;
  let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth);
  let P = unproject(ndc);

  // Lift the cone origin off the surface to avoid self-sampling the originating voxel.
  let cellSize = uGridOrigin.w;
  let origin = P + N * (cellSize * 1.5 + NORMAL_BIAS);

  // Hemisphere basis (for the AO cones below). The fill hemisphere now comes from the probe SH
  // volume; the per-pixel cone count is gone (replaced by the baked AO_CONE_COUNT for contact AO).
  let basis = build_basis(N);
  let aperture = APERTURE;
  // Per-pixel azimuth rotation of the whole cone set (interleaved-gradient-noise hash) so the
  // gaps between narrow cones fall in DIFFERENT directions each pixel → fixed "ray" banding
  // becomes fine dither, which the half-res + bilinear upsample then blurs away. Lets a narrow
  // aperture use FEWER cones without visible streaks.
  let jitter = fract(52.9829189 * fract(dot(vec2<f32>(half), vec2<f32>(0.06711056, 0.00583715)))) * 6.2831853;
  // Decorrelated [0,1) per-pixel value for radial (start-distance) dither — kills the
  // concentric "tree-ring" banding around bright sources at low cone counts.
  let jrad = fract(jitter * 1.61803399);

  // Emitter DIRECT light, SUMMED on the SAME scale as the sun (NO /wsum averaging) so a brighter
  // emitter actually competes with (and overpowers) the sun — e.g. a bright lamp near the floor
  // fills the sun-shadow it casts on itself. occAcc = AO occlusion from the short AO cones below.
  var directEmitters = vec3<f32>(0.0);
  var occAcc = 0.0;

  // (a) AIMED cones — one narrow cone per emitter, pointed straight at the light center for a
  // bright, far-reaching direct + soft-shadow term. Aperture = the light's angular size
  // (radius / distance) sets the penumbra width; the cosine term (ndl) is its weight, so these
  // fold into the SAME hemisphere integral as the fill cones below. Emitters are auto-discovered
  // (uParams2.w = live count, lights[] = world centers + radius).
  let lc = min(8, max(0, i32(uParams2.w)));
  for (var j = 0; j < lc; j = j + 1) {
    let toL = lights[j].xyz - origin;
    let d = length(toL);
    if (d < 1e-3) { continue; }
    let dir = toL / d;
    let ndl = max(dot(N, dir), 0.0);
    if (ndl <= 0.0) { continue; }
    // Distance falloff: without it the direct light is a FLAT-bright pool with a hard rim (reads as
    // an "invisible bigger sphere" the surface cuts through). atten = 1 at the light center, ~1/d²
    // far. EMITTER_FALLOFF = falloff coefficient (0 = none → flat sun-like emitter; 1 = standard);
    // lr = emitter radius (lights[j].w) so it is 1 inside the source and fades smoothly outside.
    let lr = max(lights[j].w, 1e-3);
    let atten = 1.0 / (1.0 + EMITTER_FALLOFF * (d * d) / (lr * lr));
    let full = ndl * lightColor[j].rgb * abs(lightColor[j].w) * atten;
    // LIGHT CULL: the 64-step shadow march is the dominant cost. Compute this emitter's direct
    // contribution FIRST (cheap, no trace) and skip the march entirely if it is negligible here —
    // far away, steep grazing, dim, or faded out by the falloff. For "many small emitters" most
    // pixels see only a few relevant lights, so this is a big win at ~zero visible change (the
    // skipped term was ≈0). With falloff=0 (flat sun-like) atten stays 1, so global lights are
    // never culled — exactly the intended behavior for that mode.
    if (max(full.r, max(full.g, full.b)) * EMITTER_DIRECT < 0.003) { continue; }
    let ap = clamp(lights[j].w / d, 0.02, 0.5);   // angular size of the light = penumbra width
    // AIMED: step budget + early-out opacity baked (AIMED_STEPS, AIMED_ALPHA_CUT). Defaults (32, 1.0)
    // = crisp behavior; lower steps / a <1 alphaCut trade shadow precision for speed on heavy scenes.
    let r = trace_cone(origin, dir, ap, d, 0.0, jrad, max(1, AIMED_STEPS), AIMED_ALPHA_CUT);
    // ANALYTIC DIRECT + bleed-cancel (fixes the "white shadow"). full = the emitter's own light.
    // shadow = how much the cone's opacity removes; bleed = the radiance the cone actually
    // gathered along the way (a BRIGHT occluder => big bleed => its false shadow is cancelled;
    // a DARK occluder => ~0 bleed => the shadow survives). The target emitter itself bleeds ≈ its
    // own Lj, so it always delivers full light. Clamped so it can only darken, never exceed full.
    let occ = clamp(r.a, 0.0, 1.0);
    let shadow = full * occ;
    let bleed = ndl * r.rgb;
    let contrib = full - max(vec3<f32>(0.0), shadow - bleed);
    directEmitters = directEmitters + max(vec3<f32>(0.0), contrib);
  }

  // (b) FILL / bounce — the SCREEN-SPACE probes (built in voxelScreenProbe): a bilateral 4-probe
  // resolve of the surface-anchored SH-L1 fill (see resolve_screen_probes). The low-frequency
  // hemisphere bounce, as its OWN term (scaled by GI_STRENGTH below).
  let fillAvg = resolve_screen_probes(P, N, full);

  // (c) AO — a few SHORT hemisphere occlusion cones (opacity ONLY, no radiance → no double-count
  // with the probe bounce). Becomes the .a/visibility output the composite reads as ambient
  // occlusion; kept per-pixel + short so small moving objects still darken their contacts.
  let aoCount = AO_CONE_COUNT;
  let aoReach = AO_REACH;
  let aoSteps = max(1, AO_STEPS);
  var aoW = 0.0;
  for (var a = 0; a < aoCount; a = a + 1) {
    let k = (f32(a) + 0.5) / f32(aoCount);
    let cosT = 1.0 - k;
    let sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
    let phi = f32(a) * 2.39996323 + jitter;
    let local = vec3<f32>(sinT * cos(phi), sinT * sin(phi), cosT);
    let dir = normalize(basis * local);
    let r = trace_cone(origin, dir, aperture, aoReach, 0.0, jrad, aoSteps, 0.95);
    occAcc = occAcc + cosT * r.a;
    aoW = aoW + cosT;
  }

  // Combine: emitter DIRECT (summed, sun-scale) × EMITTER_DIRECT strength + probe bounce ×
  // GI_STRENGTH. No /wsum averaging → a bright emitter competes with the sun.
  let indirect = directEmitters * EMITTER_DIRECT + fillAvg * GI_STRENGTH;
  let visibility = clamp(1.0 - occAcc / max(aoW, 1e-4), 0.0, 1.0);
  // rgb = emitter direct + indirect bounce; a = AO visibility, read by the composite.
  return vec4f(indirect, visibility);
}
`,
  );
}
