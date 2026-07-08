import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";
import { SCREEN_PROBE_K } from "./voxelResources.ts";
import { probePackWGSL } from "./voxelProbeShared.wgsl.ts";

// Adaptive screen-probe atlas — ADAPTIVE REFINE (single level, 16→8). One thread per fine cell; the
// cell center is a candidate adaptive-probe position and its parent base tile is cell / cellDiv (the
// cell divisor is a LIVE uniform — screenParams.w — so the level is GUI-tunable). The ONLY spawn
// trigger is LIGHT-ADAPTIVE: the DC-luminance SPREAD of the GATHERED incoming-irradiance SH across
// the parent's uniform cage. Where the real gathered light varies more than lightThresh across the
// sparse uniform probes, a finer probe is placed to resolve that gradient; everywhere else the
// uniform cage already represents the surface, so nothing is spawned. (The former geometric
// neighbour-agreement trigger and the second 8→4 level were removed — lightThresh alone drives it.)
//
// group 0 = uniforms (screenParams + the light-trigger params) + the G-buffer normal + the RAW
// uniform SH atlas (point-loaded for the radiometric trigger). group 1 is empty (empty-group-1
// pattern). group 2 = the StorageWrite allocator buffers (counter/data/header/indices, var<storage,
// read_write> so probeData/tileIndices are read AND written in one shader). The atomicAdds are on
// STORAGE BUFFERS (WebGPU has no storage-texture atomics).

const uC = (name: string, type: string) =>
  new VariableMeta(name, VariableKind.Uniform, type, { visibility: GPUShaderStage.COMPUTE });

const sW = (name: string, type: string, opt?: { size?: number; bufferSize?: number }) =>
  new VariableMeta(name, VariableKind.StorageWrite, type, {
    visibility: GPUShaderStage.COMPUTE,
    ...opt,
  });

export const WORKGROUP = 8; // 8*8 = 64 threads over the 2D cell grid

export const refineShaderMeta = new ShaderMeta(
  {
    // .x = canvas width (px), .y = canvas height (px), .z = SCREEN_PROBE_TILE, .w = CELL DIVISOR
    // (cellPx = tile / .w; the LIVE level knob — e.g. 2 → 8px cells at tile 16).
    screenParams: uC("screenParams", `vec4<f32>`),
    // Light-adaptive trigger params (LIVE): .x = lightThresh — the DC-luminance SPREAD of the GATHERED
    // incoming-irradiance SH across the uniform cage above which we subdivide (a real lighting
    // gradient the sparse uniform probes can't resolve). Large ⇒ effectively off (no adaptive probes).
    // .y = maxAdaptive (the hard bump-allocator budget).
    lightParams: uC("uLight", `vec4<f32>`),
    normalTex: new VariableMeta("normalTex", VariableKind.Texture, `texture_2d<f32>`, {
      visibility: GPUShaderStage.COMPUTE,
      textureSampleType: "float",
    }),
    // The RAW uniform-probe SH atlas (the SAME shR/shG/shB the UNIFORM gather wrote just before this
    // pass). Point-loaded at each valid cage probe's texel; the band-0 (DC) luminance is the real
    // incoming-irradiance signal whose spread across the cage drives the subdivision trigger.
    inShR: new VariableMeta("inShR", VariableKind.Texture, `texture_2d<f32>`, {
      visibility: GPUShaderStage.COMPUTE,
      textureSampleType: "float",
    }),
    inShG: new VariableMeta("inShG", VariableKind.Texture, `texture_2d<f32>`, {
      visibility: GPUShaderStage.COMPUTE,
      textureSampleType: "float",
    }),
    inShB: new VariableMeta("inShB", VariableKind.Texture, `texture_2d<f32>`, {
      visibility: GPUShaderStage.COMPUTE,
      textureSampleType: "float",
    }),
    // ---- group 2 : the allocator + indirection buffers (StorageWrite = read_write) ----
    // atomic types can't be parsed by getTypeSize → pass explicit size/bufferSize (smoke-test rule).
    probeCounter: sW("uCounter", `array<atomic<u32>, 2>`, { size: 2, bufferSize: 8 }),
    probeData: sW("uProbeData", `array<vec4<u32>>`),
    tileHeader: sW("uTileHeader", `array<atomic<u32>>`),
    tileIndices: sW("uTileIndices", `array<u32>`),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
const K: u32 = ${SCREEN_PROBE_K}u;
const LEVEL_TAG: u32 = 1u;     // single adaptive level → all adaptive probes are tagged level 1

${probePackWGSL}

// DC (band-0) luminance of a gathered SH-L1 probe at its atlas texel. The band-0 coefficient is
// coeff.x; multiplying by Y00 (0.282095) recovers the average radiance per channel, then a Rec.709
// luma. Its SPREAD across the uniform cage is the radiometric subdivision trigger (light-adaptive
// density) — the REAL gathered incoming irradiance.
fn probe_sh_dc_lum(texel: vec2<i32>) -> f32 {
  let dc = vec3<f32>(
    textureLoad(inShR, texel, 0).x,
    textureLoad(inShG, texel, 0).x,
    textureLoad(inShB, texel, 0).x,
  ) * 0.282095;
  return dot(dc, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let cell = vec2<i32>(gid.xy);
  let tile = i32(screenParams.z);
  let cellDiv = max(1, i32(screenParams.w)); // LIVE cell divisor (level knob): cellPx = tile/div
  let cellPx = max(1, tile / cellDiv);       // e.g. tile 16 → 8px (div 2)
  let cellsX = i32(ceil(screenParams.x / f32(cellPx)));
  let cellsY = i32(ceil(screenParams.y / f32(cellPx)));
  if (cell.x >= cellsX || cell.y >= cellsY) { return; }

  let gw = i32(ceil(screenParams.x / screenParams.z));
  let gh = i32(ceil(screenParams.y / screenParams.z));
  let parent = cell / cellDiv;              // the candidate's coarse base tile
  if (parent.x < 0 || parent.y < 0 || parent.x >= gw || parent.y >= gh) { return; }
  let numUniform = gw * gh;
  let parentSlot = parent.y * gw + parent.x;

  let dimsI = vec2<i32>(screenParams.xy) - vec2<i32>(1);
  let candidate = min(cell * cellPx + vec2<i32>(cellPx / 2), dimsI);

  // Skip the one cell that coincides with the parent tile's uniform representative pixel (already
  // covered by the uniform probe → spawning there would be a duplicate).
  let reprPix = sp_unpack_pixel(uProbeData[parentSlot]);
  let reprCell = reprPix / cellPx;
  if (reprCell.x == cell.x && reprCell.y == cell.y) { return; }

  // No surface at the candidate (sky) → never spawn (adaptive probes only sit on geometry).
  if (textureLoad(normalTex, candidate, 0).a < 0.5) { return; }

  // The 4 surrounding UNIFORM probes (the bilinear cage the resolve uses): probe c sits at pixel
  // c*tile + tile/2 → grid float coord gf = candidate/tile - 0.5, corners g0 + {(0,0)..(1,1)}.
  // Min/max DC-luminance of the gathered SH across the VALID cage → the radiometric spread.
  let gf = vec2<f32>(candidate) / screenParams.z - vec2<f32>(0.5);
  let g0 = vec2<i32>(floor(gf));
  var lmin = 1e30;
  var lmax = -1e30;
  var lany = false;
  for (var q = 0; q < 4; q = q + 1) {
    let off = vec2<i32>(q & 1, (q >> 1) & 1);
    let c = g0 + off;
    if (c.x < 0 || c.y < 0 || c.x >= gw || c.y >= gh) { continue; }
    let cSlot = c.y * gw + c.x;
    // A cage probe only counts toward the spread if it sits on a surface (a sky probe's SH is 0,
    // which would falsely inflate the spread on silhouettes and force a spurious light-spawn there).
    if (textureLoad(normalTex, sp_unpack_pixel(uProbeData[cSlot]), 0).a < 0.5) { continue; }
    // The uniform block is identity-mapped into atlas rows [0, gh), so a uniform probe at grid coord
    // c lives at atlas texel c. Sample its gathered-SH DC luminance for the radiometric spread.
    let lum = probe_sh_dc_lum(c);
    lmin = min(lmin, lum);
    lmax = max(lmax, lum);
    lany = true;
  }

  // LIGHT-ADAPTIVE trigger: subdivide only where the real gathered incoming light varies across the
  // cage by more than lightThresh. lany guards the degenerate all-invalid cage; lightThresh large ⇒
  // off (no adaptive probes at all → the flat uniform atlas).
  let lightThresh = uLight.x;
  if (!(lany && (lmax - lmin) > lightThresh)) { return; }

  // Bump-allocate an adaptive probe. Overflow ⇒ degrade (skip + sticky flag), never corrupt a slot.
  let maxAdaptive = u32(max(0.0, uLight.y));
  let slot = atomicAdd(&uCounter[0], 1u);
  if (slot >= maxAdaptive) {
    atomicStore(&uCounter[1], 1u);
    return;
  }
  let g = u32(numUniform) + slot;
  // Store the repr pixel + level tag + the probe FOOTPRINT (cellPx, its cell size in full-res px) in
  // .z. The footprint area-weights the probe in the resolve so N of these fine adaptive probes weigh
  // the same collectively as the 1 coarse uniform probe they subdivide (density-invariant average).
  var rec = sp_pack_probe(candidate, LEVEL_TAG);         // non-atomic: this slot is unique to this thread
  rec.z = u32(cellPx);
  uProbeData[g] = rec;
  let j = atomicAdd(&uTileHeader[parentSlot], 1u);
  if (j < K) {                                           // else: allocated but unreachable from resolve — ok
    uTileIndices[u32(parentSlot) * K + j] = g;
  }
}
`,
);
