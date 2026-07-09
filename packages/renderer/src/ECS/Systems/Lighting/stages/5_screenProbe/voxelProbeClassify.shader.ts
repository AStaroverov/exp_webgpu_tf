import { VariableKind, VariableMeta } from "../../../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../../../WGSL/wgsl.ts";
import { probePackWGSL } from "../../core/shaders/voxelProbeShared.wgsl.ts";

// Adaptive screen-probe atlas — PASS A0 (uniform placement). One thread per COARSE 16px tile picks
// the tile's representative full-res pixel (tile center + a stable per-tile jitter, exactly as the
// flat-atlas gather did inline) and writes it into probeData[tileIdx] at level 0. It does NO tracing
// and reads NO G-buffer — it only decides WHERE each uniform probe sits, so the gather (indirect,
// later pass) can read probeData[slot] instead of recomputing the placement, and the refine pass
// (A1) can read the uniform probes' positions to run its neighbor-agreement test. Validity (sky /
// no-surface) is re-derived from the G-buffer normal by the gather + refine at the stored pixel.
//
// group 0 = screenParams uniform. group 2 = probeData (StorageWrite). group 1 is empty (empty-
// group-1 pattern). tileHeader is NOT touched here — the per-frame clearBuffer already zeroes it.

const uC = (name: string, type: string) =>
  new VariableMeta(name, VariableKind.Uniform, type, { visibility: GPUShaderStage.COMPUTE });

export const WORKGROUP = 8; // 8*8 = 64 threads over the 2D coarse-tile grid

export const classifyShaderMeta = new ShaderMeta(
  {
    // .x = canvas width (px), .y = canvas height (px), .z = SCREEN_PROBE_TILE (full-res px / probe),
    // .w spare.
    screenParams: uC("screenParams", `vec4<f32>`),
    // ---- group 2 : probeData (StorageWrite = var<storage, read_write>), one vec4<u32> per probe ----
    probeData: new VariableMeta("uProbeData", VariableKind.StorageWrite, `array<vec4<u32>>`, {
      visibility: GPUShaderStage.COMPUTE,
    }),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
${probePackWGSL}

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let coord = vec2<i32>(gid.xy);
  let tile = i32(screenParams.z);
  // Coarse-grid dims = ceil(canvas / tile). Dispatch is ceil-rounded; drop threads past them.
  let gw = i32(ceil(screenParams.x / screenParams.z));
  let gh = i32(ceil(screenParams.y / screenParams.z));
  if (coord.x >= gw || coord.y >= gh) {
    return;
  }

  // Uniform probe's global slot = the identity-mapped atlas position (see voxelResources).
  let slot = coord.y * gw + coord.x;

  // Representative full-res pixel = tile center + a STABLE per-tile spatial jitter (hashed from
  // coord → the SAME every frame, so the probe placement never flickers). Kept within ±tile/4 of
  // the center; clamped into the canvas. (Adaptive subdivision — pass A1 — fills the surfaces this
  // single center probe cannot represent, so no per-tile "snap to nearest surface" pick is needed.)
  let h = fract(52.9829189 * fract(dot(vec2<f32>(coord), vec2<f32>(0.06711056, 0.00583715))));
  let jit = vec2<i32>((vec2<f32>(h, fract(h * 1.61803399)) - vec2<f32>(0.5)) * screenParams.z * 0.5);
  let center = coord * tile + vec2<i32>(tile / 2) + jit;
  let dimsI = vec2<i32>(screenParams.xy) - vec2<i32>(1);
  let full = min(center, dimsI);

  // Store the representative pixel + level 0 + the probe FOOTPRINT (its cell size in full-res px =
  // the coarse tile for a uniform probe) in .z. The footprint area-weights the probe in the resolve
  // so N fine adaptive probes collectively weigh the same as the 1 coarse uniform probe they
  // subdivide (density-invariant average). Validity is re-derived from the G-buffer normal by the
  // gather (n.a < 0.5 → invalid) and by the refine pass, so it is not packed here.
  var rec = sp_pack_probe(full, 0u);
  rec.z = u32(tile);
  uProbeData[slot] = rec;
}
`,
);
