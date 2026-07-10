import { VariableKind, VariableMeta } from "../../../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../../../WGSL/wgsl.ts";
import { probePackWGSL, probeRingWGSL } from "../../core/shaders/voxelProbeShared.wgsl.ts";
import { probeTile, ringThresholds, type VoxelBakedConfig } from "../../core/voxelConfig.ts";

// Adaptive screen-probe atlas — REFINEMENT DECISION (per coarse tile). One thread per tile decides
// whether the WHOLE tile is refined, and integrates that decision through a persistent hysteresis
// counter. The refine pass then spawns ALL of an active tile's fine cells (a locally-uniform fine
// lattice) and the resolve EXCLUDES the tile's uniform probe — the dense lattice REPLACES the
// sparse one instead of overlaying it (a refined tile is statistically identical to a patch of a
// plain fine-tile grid; the old per-cell spawning produced a coarse-jittered probe + N cell-center
// probes + a hole at the repr cell — a mongrel lattice that resolved visibly differently from the
// uniform-fine reference).
//
// RAW trigger: the DC-luminance spread of the GATHERED uniform SH across the tile's 3×3
// neighborhood (both tiles adjacent to a lighting gradient refine — a 2×2 cage would refine only
// one side). The raw signal is temporally noisy (stochastic cone sampling rides the SH), so it
// only steps the persistent per-tile counter: +2 on trigger (cap 15), −1 otherwise; the tile is
// REFINED while the counter ≥ 8. One thread owns each tile → plain (non-atomic) RMW is race-free,
// and the refine pass reads the settled state across a pass barrier.
//
// group 0 = uniforms + G-buffer normal + the RAW uniform SH atlas (this frame's write set —
// gatherUniform ran just before). group 1 = probeData (READ — cage validity needs the repr
// pixels). group 2 = refineState (read_write).

const uC = (name: string, type: string) =>
  new VariableMeta(name, VariableKind.Uniform, type, { visibility: GPUShaderStage.COMPUTE });

export const DECIDE_WORKGROUP = 8; // 8*8 = 64 threads over the 2D coarse-tile grid

export const createDecideShaderMeta = (cfg: VoxelBakedConfig) => new ShaderMeta(
  {
    // Canvas dims (px) — the only live value (the tile / trigger threshold / ring thresholds are
    // BAKED consts — SP_TILE / LIGHT_THRESH / RING_R0 / RING_R1).
    screenParams: uC("screenParams", `vec2<f32>`),
    normalTex: new VariableMeta("normalTex", VariableKind.Texture, `texture_2d<f32>`, {
      visibility: GPUShaderStage.COMPUTE,
      textureSampleType: "float",
    }),
    // The RAW uniform SH atlas (what gatherUniform wrote THIS frame). Point-loaded; band-0 (DC)
    // luminance is the incoming-irradiance signal whose spread drives the decision.
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
    // ---- group 1 : probeData (StorageRead) — repr pixels for the cage validity test ----
    probeData: new VariableMeta("uProbeData", VariableKind.StorageRead, `array<vec4<u32>>`, {
      visibility: GPUShaderStage.COMPUTE,
    }),
    // ---- group 2 : the PERSISTENT per-tile hysteresis counter (StorageWrite = read_write) ----
    refineState: new VariableMeta("uRefineState", VariableKind.StorageWrite, `array<u32>`, {
      visibility: GPUShaderStage.COMPUTE,
    }),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
${probePackWGSL}
${probeRingWGSL}

// DC (band-0) luminance of a gathered SH-L1 probe at its atlas texel: coeff.x × Y00 (0.282095)
// recovers the average radiance per channel, then a Rec.709 luma.
fn probe_sh_dc_lum(texel: vec2<i32>) -> f32 {
  let dc = vec3<f32>(
    textureLoad(inShR, texel, 0).x,
    textureLoad(inShG, texel, 0).x,
    textureLoad(inShB, texel, 0).x,
  ) * 0.282095;
  return dot(dc, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Membership hysteresis: up-fast (+2, cap 15) / down-slow (−1); the tile is REFINED while ≥ 8.
// A solid trigger activates in ~4 frames; a borderline 50/50 flicker drifts UP (net +0.5/frame)
// and stays refined instead of blinking; full deactivation takes ≥ 8 quiet frames.
const HYST_MAX: u32 = 15u;
const HYST_UP: u32 = 2u;

// BAKED trigger + foveated ring thresholds (see voxelConfig.ts). LIGHT_THRESH large ⇒ never
// triggers ⇒ every counter decays to 0 ⇒ the flat uniform lattice.
const SP_TILE: f32 = ${probeTile(cfg)};
const LIGHT_THRESH: f32 = ${cfg.lightThresh};
const RING_R0: f32 = ${ringThresholds(cfg).r0};
const RING_R1: f32 = ${ringThresholds(cfg).r1};

@compute @workgroup_size(${DECIDE_WORKGROUP}, ${DECIDE_WORKGROUP}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tileC = vec2<i32>(gid.xy);
  let gw = i32(ceil(screenParams.x / SP_TILE));
  let gh = i32(ceil(screenParams.y / SP_TILE));
  if (tileC.x >= gw || tileC.y >= gh) { return; }

  // FOVEATED rings: the persistent state is owned per BASE-level block, by its anchor tile — only
  // that thread proceeds (one writer per counter, race-free at every level). At level 0 (center /
  // rings off) every tile is its own anchor — the original per-tile behavior.
  let sb = sp_ring_base_level(tileC, screenParams.xy, SP_TILE, RING_R0, RING_R1);
  let sbs = 1 << u32(sb);
  if (any((tileC & vec2<i32>(sbs - 1)) != vec2<i32>(0))) { return; }
  let slot = tileC.y * gw + tileC.x;

  // RAW trigger: DC-luminance spread across the 3×3 SAME-LEVEL neighborhood (stride = the base
  // block size, so a ring block compares against its equal-granularity neighbors). A neighbor only
  // counts if it actually holds a probe this frame (footprint > 0 — anchors nest, but a coarser
  // neighboring region leaves non-anchor tiles empty) AND sits on a surface (a sky probe's SH is
  // 0 — it would falsely inflate the spread on silhouettes and refine every skyline tile).
  var lmin = 1e30;
  var lmax = -1e30;
  var lany = false;
  for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      let c = tileC + vec2<i32>(dx, dy) * sbs;
      if (c.x < 0 || c.y < 0 || c.x >= gw || c.y >= gh) { continue; }
      let cSlot = c.y * gw + c.x;
      let rec = uProbeData[cSlot];
      if (rec.z == 0u) { continue; }
      if (textureLoad(normalTex, sp_unpack_pixel(rec), 0).a < 0.5) { continue; }
      // The uniform block is identity-mapped into atlas rows [0, gh) → texel = grid coord.
      let lum = probe_sh_dc_lum(c);
      lmin = min(lmin, lum);
      lmax = max(lmax, lum);
      lany = true;
    }
  }
  let raw = lany && (lmax - lmin) > LIGHT_THRESH;

  var st = uRefineState[slot];
  if (raw) {
    st = min(st + HYST_UP, HYST_MAX);
  } else if (st > 0u) {
    st = st - 1u;
  }
  uRefineState[slot] = st;
}
`,
);
