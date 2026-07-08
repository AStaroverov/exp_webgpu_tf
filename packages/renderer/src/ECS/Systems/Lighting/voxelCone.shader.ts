import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";
import { VoxelBakedConfig } from "./voxelConfig.ts";
import { SCREEN_PROBE_K } from "./voxelResources.ts";
import { probeWeightWGSL } from "./voxelProbeShared.wgsl.ts";

// VCT Layer 3 — SCREEN-PROBE RESOLVE + CONTACT AO. A fullscreen pass over the G-buffer:
//   1. Reconstruct the per-pixel world position P (from the reverse-Z depth + invViewProj)
//      and world normal N (from the normal G-buffer; n.rgb*2-1, n.a<0.5 = no surface).
//   2. Resolve the SCREEN-PROBE SH-L1 field at (P, N) — resolve_screen_probes: a unified
//      bilateral multi-probe average over the uniform grid + adaptive probes, with a loose
//      spatial fallback on disocclusion. ALL cone tracing (fill hemisphere AND the aimed
//      emitter cones) lives in the probe gather (voxelScreenProbe.shader.ts) and is temporally
//      amortized there; this pass only reads the resulting SH atlas — its per-pixel cost no
//      longer scales with light count or cone budget.
//   3. Trace a few SHORT per-pixel AO cones (opacity only, isotropic pyramid) for near-field
//      contact occlusion → the .a/visibility output.
//   4. Write rgb = resolved indirect (probe fill + emitter light, × giStrength), a = AO.
//
// PERF: this pass runs at a DOWNSCALED resolution (half by default → ¼ the pixels; quarter →
// 1/16). The downscale factor is chosen on the CPU via the cone output size; the shader is
// resolution-agnostic (maps via texCoord). The composite normal-aware-upsamples this output
// back to full res — indirect light is low-frequency, so that is fine.
//
// The AO cones read only the isotropic voxelRadiance pyramid: their reach is a couple of cells,
// so the anisotropic far-field anti-leak never mattered for them — the 6 directional volumes are
// bound (and sampled) exclusively by the probe gather, where the long cones live.
//
// Fullscreen pass with `unproject` + reverse-Z NDC reconstruction. textureSampleLevel (explicit
// LOD) is used in the trace loop — legal in non-uniform control flow (unlike textureSample).

export function createConeShaderMeta(cfg: VoxelBakedConfig) {
  return new ShaderMeta(
  {
    // .x = screen width (px), .y = screen height (px), .zw spare (the aniso toggle + light count
    // that used to ride here moved to the probe gather's uLightParams — this pass keeps only the
    // probe resolve + iso AO cones).
    params2: new VariableMeta("uParams2", VariableKind.Uniform, `vec4<f32>`),
    // Screen-probe resolve params (all LIVE per-frame uniforms — GUI-tunable with no rebuild):
    // .x = SCREEN_PROBE_TILE (full-res px / probe), .y = normal-weight power (SP_NORMAL_POW),
    // .z = plane-threshold scale (SP_PLANE_K, × local probe spacing), .w = resolveRadius (the smooth
    // screen kernel's support in TILES — bigger = smoother/wider fill, smaller = more local detail).
    params3: new VariableMeta("uParams3", VariableKind.Uniform, `vec4<f32>`),
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
    // The voxelRadiance mip pyramid (ALL mips) — read at the per-step LOD by the short AO cones.
    voxelRadiance: new VariableMeta("voxelRadiance", VariableKind.Texture, `texture_3d<f32>`, {
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
    // Filtering sampler for textureSampleLevel over the voxelRadiance pyramid.
    voxelSampler: new VariableMeta("voxelSampler", VariableKind.Sampler, `sampler`),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
// BAKED tuning consts (interpolated from VoxelBakedConfig at shader-build time). See voxelConfig.ts.
const NORMAL_BIAS: f32 = ${cfg.normalBias};
const APERTURE: f32 = ${cfg.aperture};
const GI_STRENGTH: f32 = ${cfg.giStrength};
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

// One SHORT AO cone marched along dir from origin: diameter grows with distance, each step samples
// the isotropic voxelRadiance pyramid at LOD = log2(diameter/voxelSize) and composites opacity
// front-to-back ("over"). Only .a (occlusion) is consumed — the radiance term is unused; the
// diffuse fill comes entirely from the probe resolve. The step is floored at reach/maxSteps so the
// cone always spans its full reach within the step budget. startJ ∈ [0,1) scales the start distance
// per pixel — the ~geometric march makes that a per-pixel shift of the sampling shells at EVERY
// distance, dithering the concentric "tree-ring" banding away under the half-res upsample.
fn trace_cone(origin: vec3<f32>, dir: vec3<f32>, aperture: f32, reach: f32, startJ: f32, maxSteps: i32, alphaCut: f32) -> vec4<f32> {
  var col = vec3<f32>(0.0);
  var alpha = 0.0;
  let voxelSize = uGridOrigin.w;
  let gridMin = uGridOrigin.xyz;
  let extent = vec3<f32>(uGridDims.xyz) * voxelSize;
  let stepFloor = reach / f32(maxSteps);
  var dist = voxelSize * (1.0 + startJ);
  for (var i = 0; i < maxSteps; i = i + 1) {
    if (alpha >= alphaCut || dist > reach) { break; }
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

  // Hemisphere basis (for the AO cones below). The fill hemisphere AND the emitter light come from
  // the probe SH volume (traced + temporally amortized in voxelScreenProbe).
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

  // occAcc = AO occlusion from the short AO cones below.
  var occAcc = 0.0;

  // (a) FILL / bounce — the SCREEN-SPACE probes (built in voxelScreenProbe): a bilateral multi-probe
  // resolve of the surface-anchored SH-L1 fill (see resolve_screen_probes). Carries the emitter
  // direct term too (the aimed cones are traced per probe in the gather).
  let fillAvg = resolve_screen_probes(P, N, full);

  // (b) AO — a few SHORT hemisphere occlusion cones (opacity ONLY, no radiance → no double-count
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
    let r = trace_cone(origin, dir, aperture, aoReach, jrad, aoSteps, 0.95);
    occAcc = occAcc + cosT * r.a;
    aoW = aoW + cosT;
  }

  // Combine: probe bounce × GI_STRENGTH. The probe SH already carries the emitter direct term
  // (scaled by EMITTER_DIRECT at the probe gather), so fillAvg is the whole indirect+emitter light.
  let indirect = fillAvg * GI_STRENGTH;
  let visibility = clamp(1.0 - occAcc / max(aoW, 1e-4), 0.0, 1.0);
  // rgb = emitter direct + indirect bounce; a = AO visibility, read by the composite.
  return vec4f(indirect, visibility);
}
`,
  );
}
