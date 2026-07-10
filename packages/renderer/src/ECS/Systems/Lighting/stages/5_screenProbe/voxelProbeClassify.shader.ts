import { VariableKind, VariableMeta } from "../../../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../../../WGSL/wgsl.ts";
import { probePackWGSL, probeRingWGSL } from "../../core/shaders/voxelProbeShared.wgsl.ts";
import { probeTile, ringThresholds, type VoxelBakedConfig } from "../../core/voxelConfig.ts";

// Adaptive screen-probe atlas — PASS A0 (uniform placement). One thread per COARSE 16px tile picks
// the tile's representative full-res pixel (tile center + a stable per-tile jitter, exactly as the
// flat-atlas gather did inline) and writes it into probeData[tileIdx] at level 0. It does NO tracing
// and reads NO G-buffer — it only decides WHERE each uniform probe sits, so the gather (indirect,
// later pass) can read probeData[slot] instead of recomputing the placement, and the refine pass
// (A1) can read the uniform probes' positions to run its neighbor-agreement test. Validity (sky /
// no-surface) is re-derived from the G-buffer normal by the gather + refine at the stored pixel.
//
// FOVEATED RINGS: the base lattice thins out with radial screen-center distance (level 0/1/2 →
// one probe per 1/4/16 tiles; see probeRingWGSL). Only the BLOCK ANCHOR tile of its level places a
// probe (block center + jitter scaled to the block, footprint = block px); every other tile stores
// an EMPTY record (footprint 0) — the gather writes it out as invalid, the resolve skips it. The
// persistent refineState (written by decide LAST frame — one frame of lag, smoothed by the
// hysteresis) BOOSTS an active block one level finer, so a lighting gradient in a sparse ring gets
// its density back by the same ×4 multiplier fine refinement uses in the center.
//
// group 0 = screenParams uniform (rings are baked consts). group 1 = refineState (StorageRead — the boost).
// group 2 = probeData (StorageWrite). tileHeader is NOT touched here — the per-frame clearBuffer
// already zeroes it.

const uC = (name: string, type: string) =>
  new VariableMeta(name, VariableKind.Uniform, type, { visibility: GPUShaderStage.COMPUTE });

export const WORKGROUP = 8; // 8*8 = 64 threads over the 2D coarse-tile grid

export const createClassifyShaderMeta = (cfg: VoxelBakedConfig) => new ShaderMeta(
  {
    // Canvas dims (px) — the only live value (the tile is the BAKED SP_TILE; the ring thresholds
    // are RING_R0/RING_R1).
    screenParams: uC("screenParams", `vec2<f32>`),
    // ---- group 1 : the persistent per-block hysteresis state (READ — decide wrote it last frame) ----
    refineState: new VariableMeta("uRefineState", VariableKind.StorageRead, `array<u32>`, {
      visibility: GPUShaderStage.COMPUTE,
    }),
    // ---- group 2 : probeData (StorageWrite = var<storage, read_write>), one vec4<u32> per probe ----
    probeData: new VariableMeta("uProbeData", VariableKind.StorageWrite, `array<vec4<u32>>`, {
      visibility: GPUShaderStage.COMPUTE,
    }),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
${probePackWGSL}
${probeRingWGSL}

// BAKED probe lattice pitch (full-res px / probe — also sizes the CPU atlas from the same config)
// + the foveated ring thresholds (rings off ⇒ 9 = past any on-screen radius ⇒ level 0 everywhere).
const SP_TILE: f32 = ${probeTile(cfg)};
const RING_R0: f32 = ${ringThresholds(cfg).r0};
const RING_R1: f32 = ${ringThresholds(cfg).r1};

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let coord = vec2<i32>(gid.xy);
  let tile = i32(SP_TILE);
  // Coarse-grid dims = ceil(canvas / tile). Dispatch is ceil-rounded; drop threads past them.
  let gw = i32(ceil(screenParams.x / SP_TILE));
  let gh = i32(ceil(screenParams.y / SP_TILE));
  if (coord.x >= gw || coord.y >= gh) {
    return;
  }

  // Uniform probe's global slot = the identity-mapped atlas position (see voxelResources).
  let slot = coord.y * gw + coord.x;

  // FOVEATED base level from the tile's screen-center radius, BOOSTED one step finer while the
  // base block's persistent refine state is active (the state lives at the BASE anchor's slot).
  let sb = sp_ring_base_level(coord, screenParams.xy, SP_TILE, RING_R0, RING_R1);
  var lv = sb;
  if (sb > 0) {
    let ba = sp_ring_anchor(coord, sb);
    if (uRefineState[u32(ba.y * gw + ba.x)] >= SP_RING_HYST_ACTIVE) { lv = sb - 1; }
  }
  let s = 1 << u32(lv);
  // Only the block's anchor tile carries a probe; the rest store an EMPTY record (footprint 0 —
  // the gather marks the texel invalid, the resolve's validity/area tests skip it).
  if (any((coord & vec2<i32>(s - 1)) != vec2<i32>(0))) {
    uProbeData[slot] = vec4<u32>(0u);
    return;
  }

  // Representative full-res pixel = BLOCK center + a STABLE per-block spatial jitter (hashed →
  // the SAME every frame, so the probe placement never flickers). Kept within ±block/4 of the
  // center (the exact tile-level recipe scaled by the block size — a coarse ring must be
  // statistically identical to a plain grid at tile = blockPx); clamped into the canvas.
  // The hash input is the BLOCK coord (anchor >> level), NOT the anchor tile coord: the IGN hash
  // decorrelates only unit-step inputs — sampled at stride 2/4 (ring anchors) its values ramp
  // linearly and the jittered dots line up in DIAGONAL chains across the ring. Block space steps
  // by 1 at every level, so each ring jitters like a plain level-0 grid.
  let blockPx = tile * s;
  let bc = coord >> vec2<u32>(u32(lv));
  let h = fract(52.9829189 * fract(dot(vec2<f32>(bc), vec2<f32>(0.06711056, 0.00583715))));
  let jit = vec2<i32>((vec2<f32>(h, fract(h * 1.61803399)) - vec2<f32>(0.5)) * f32(blockPx) * 0.5);
  let center = coord * tile + vec2<i32>(blockPx / 2) + jit;
  let dimsI = vec2<i32>(screenParams.xy) - vec2<i32>(1);
  let full = clamp(center, vec2<i32>(0), dimsI);

  // Store the representative pixel + level 0 + the probe FOOTPRINT (its cell size in full-res px =
  // the block for a ring probe, the coarse tile at level 0) in .z. The footprint area-weights the
  // probe in the resolve so density changes never shift brightness (density-invariant average).
  // Validity is re-derived from the G-buffer normal by the gather (n.a < 0.5 → invalid).
  var rec = sp_pack_probe(full, 0u);
  rec.z = u32(blockPx);
  uProbeData[slot] = rec;
}
`,
);
