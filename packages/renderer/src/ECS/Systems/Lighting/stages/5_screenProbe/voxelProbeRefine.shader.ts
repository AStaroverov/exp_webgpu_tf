import { VariableKind, VariableMeta } from "../../../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../../../WGSL/wgsl.ts";
import { SCREEN_PROBE_K } from "../../core/voxelResources.ts";
import { probePackWGSL, probeRingWGSL } from "../../core/shaders/voxelProbeShared.wgsl.ts";
import {
  probeRefineDiv,
  probeTile,
  ringThresholds,
  type VoxelBakedConfig,
} from "../../core/voxelConfig.ts";

// Adaptive screen-probe atlas — REFINE (spawn). One thread per fine cell (cellPx = tile/cellDiv,
// the LIVE divisor — screenParams.w). The refinement DECISION is not made here: the probeDecide
// pass (one thread per coarse tile + persistent hysteresis) settles which tiles are refined; this
// pass reads that state across the pass barrier and spawns a probe in EVERY fine cell of an
// active tile (sky cells excepted). The refined tile is thus a locally-uniform fine lattice —
// cell center + the SAME stable per-cell hash jitter the uniform classify uses, so a refined
// patch is statistically identical to a plain fine-tile uniform grid — and the resolve EXCLUDES
// the tile's uniform probe (tileHeader > 0), so the dense lattice REPLACES the sparse one instead
// of overlaying it.
//
// STABLE IDENTITY: an active cell's probe slot is DIRECT-MAPPED (numUniform + fineIdx), so the
// same cell writes the same probeData entry and the same atlas texel every frame → the gather can
// find its OWN reprojected temporal history (bump-allocated slots shuffled per frame and could
// never accumulate). The bump counter only builds the per-frame ACTIVE WORK LIST for the indirect
// gather dispatch — its order feeds nothing but thread→slot mapping.
//
// group 0 = uniforms + the G-buffer normal (sky test). group 1 = refineState (READ — written by
// probeDecide). group 2 = the StorageWrite allocator buffers (counter/data/header/indices/
// activeList). The atomicAdds are on STORAGE BUFFERS (WebGPU has no storage-texture atomics).

const uC = (name: string, type: string) =>
  new VariableMeta(name, VariableKind.Uniform, type, { visibility: GPUShaderStage.COMPUTE });

const sW = (name: string, type: string, opt?: { size?: number; bufferSize?: number }) =>
  new VariableMeta(name, VariableKind.StorageWrite, type, {
    visibility: GPUShaderStage.COMPUTE,
    ...opt,
  });

export const WORKGROUP = 8; // 8*8 = 64 threads over the 2D cell grid

export const createRefineShaderMeta = (cfg: VoxelBakedConfig) => new ShaderMeta(
  {
    // .x/.y = canvas dims (px), .z = maxAdaptive (the hard bump-allocator budget — CANVAS-derived,
    // so it stays live), .w spare. (The tile + cell divisor are the BAKED SP_TILE / CELL_DIV —
    // they also size the CPU atlas from the same config; the light trigger lives in probeDecide.)
    screenParams: uC("screenParams", `vec4<f32>`),
    normalTex: new VariableMeta("normalTex", VariableKind.Texture, `texture_2d<f32>`, {
      visibility: GPUShaderStage.COMPUTE,
      textureSampleType: "float",
    }),
    // ---- group 1 : the per-tile hysteresis state (StorageRead — probeDecide wrote it) ----
    refineState: new VariableMeta("uRefineState", VariableKind.StorageRead, `array<u32>`, {
      visibility: GPUShaderStage.COMPUTE,
    }),
    // ---- group 2 : the allocator + indirection buffers (StorageWrite = read_write) ----
    // atomic types can't be parsed by getTypeSize → pass explicit size/bufferSize (smoke-test rule).
    probeCounter: sW("uCounter", `array<atomic<u32>, 2>`, { size: 2, bufferSize: 8 }),
    probeData: sW("uProbeData", `array<vec4<u32>>`),
    tileHeader: sW("uTileHeader", `array<atomic<u32>>`),
    tileIndices: sW("uTileIndices", `array<u32>`),
    // THIS frame's active adaptive slots (the gather work list), filled in bump order.
    activeList: sW("uActiveList", `array<u32>`),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
const K: u32 = ${SCREEN_PROBE_K}u;
const LEVEL_TAG: u32 = 1u;     // single adaptive level → all adaptive probes are tagged level 1

${probePackWGSL}
${probeRingWGSL}

// BAKED lattice pitch + cell divisor (cellPx = SP_TILE / CELL_DIV) + the foveated ring thresholds
// (fine sub-tile refinement exists only in the level-0 center).
const SP_TILE: f32 = ${probeTile(cfg)};
const CELL_DIV: i32 = ${probeRefineDiv(cfg)};
const RING_R0: f32 = ${ringThresholds(cfg).r0};
const RING_R1: f32 = ${ringThresholds(cfg).r1};

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let cell = vec2<i32>(gid.xy);
  let tile = i32(SP_TILE);
  let cellDiv = CELL_DIV;                    // baked cell divisor: cellPx = tile/div
  let cellPx = max(1, tile / cellDiv);       // e.g. tile 16 → 8px (div 2)
  let cellsX = i32(ceil(screenParams.x / f32(cellPx)));
  let cellsY = i32(ceil(screenParams.y / f32(cellPx)));
  if (cell.x >= cellsX || cell.y >= cellsY) { return; }

  let gw = i32(ceil(screenParams.x / SP_TILE));
  let gh = i32(ceil(screenParams.y / SP_TILE));
  let parent = cell / cellDiv;              // this cell's coarse base tile
  if (parent.x < 0 || parent.y < 0 || parent.x >= gw || parent.y >= gh) { return; }
  let numUniform = gw * gh;
  let parentSlot = parent.y * gw + parent.x;

  // FOVEATED rings: fine (sub-tile) refinement exists only in the level-0 CENTER region. In the
  // sparse rings the SAME active state instead restores one lattice step in classify (block →
  // sub-block), so "refine" is a uniform ×4-density step everywhere — just realized by different
  // machinery per ring.
  if (sp_ring_base_level(parent, screenParams.xy, SP_TILE, RING_R0, RING_R1) != 0) {
    return;
  }

  // The tile-level refinement decision (probeDecide's persistent hysteresis counter).
  if (uRefineState[parentSlot] < SP_RING_HYST_ACTIVE) { return; }

  // Fixed fine-lattice index — the cell's PERMANENT identity. Stride = the CAPACITY grid (gw·div,
  // what the buffers/atlas are sized for), not the ceil-rounded live cell count, so the mapping
  // never shifts with canvas rounding. Its direct-mapped slot is numUniform + fineIdx.
  let fineStride = gw * cellDiv;
  let fineIdx = cell.y * fineStride + cell.x;

  // Representative pixel = cell center + the SAME stable per-cell hash jitter the uniform classify
  // uses (scaled to the fine cell: ±cellPx/4) — a refined patch must be statistically identical to
  // a plain uniform grid at tile = cellPx, or the density seam itself becomes visible. The jitter
  // is hashed from the cell coord → identical every frame (placement never flickers).
  let h = fract(52.9829189 * fract(dot(vec2<f32>(cell), vec2<f32>(0.06711056, 0.00583715))));
  let jit = vec2<i32>((vec2<f32>(h, fract(h * 1.61803399)) - vec2<f32>(0.5)) * f32(cellPx) * 0.5);
  let dimsI = vec2<i32>(screenParams.xy) - vec2<i32>(1);
  let candidate = clamp(cell * cellPx + vec2<i32>(cellPx / 2) + jit, vec2<i32>(0), dimsI);

  // Sky cells never spawn (no surface to anchor to; the gather would only mark them invalid —
  // don't waste gather budget on them).
  if (textureLoad(normalTex, candidate, 0).a < 0.5) { return; }

  // Append to this frame's gather work list (budget-capped; overflow ⇒ degrade with the sticky
  // flag, never corrupt: an ungathered probe must not be published to probeData/tileIndices — the
  // resolve may only reach slots whose SH the gather wrote this frame).
  let maxAdaptive = u32(max(0.0, screenParams.z));
  let li = atomicAdd(&uCounter[0], 1u);
  if (li >= maxAdaptive) {
    atomicStore(&uCounter[1], 1u);
    return;
  }
  let fineSlot = u32(numUniform + fineIdx);
  uActiveList[li] = fineSlot;
  // Store the repr pixel + level tag + the probe FOOTPRINT (cellPx, its cell size in full-res px) in
  // .z. The footprint area-weights the probe in the resolve so the div² fine probes collectively
  // weigh the same as the 1 uniform probe they REPLACE (density-invariant average).
  var rec = sp_pack_probe(candidate, LEVEL_TAG); // non-atomic: this slot is unique to this thread
  rec.z = u32(cellPx);
  uProbeData[fineSlot] = rec;
  let j = atomicAdd(&uTileHeader[parentSlot], 1u);
  if (j < K) {                                   // else: allocated but unreachable from resolve — ok
    uTileIndices[u32(parentSlot) * K + j] = fineSlot;
  }
}
`,
);
