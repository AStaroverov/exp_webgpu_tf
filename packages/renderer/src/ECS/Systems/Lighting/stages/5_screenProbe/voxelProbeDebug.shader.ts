import { VariableKind, VariableMeta } from "../../../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../../../WGSL/wgsl.ts";
import { SCREEN_PROBE_K } from "../../core/voxelResources.ts";
import { probeTile, ringThresholds, type VoxelBakedConfig } from "../../core/voxelConfig.ts";

// Screen-probe DEBUG visualization — a fullscreen pass that replaces the composite (when the GUI
// "debug: probe layers" toggle is on) so you can SEE how the adaptive probes are distributed:
//   • faint gray  = geometry (context, from the G-buffer normal) + a faint coarse-tile grid.
//   • red tint    = coarse tiles that spawned ≥1 adaptive probe (heat by count / K) — i.e. where
//                   the refine pass decided the uniform grid could not represent the surface
//                   (silhouettes / thin features / depth edges).
//   • GREEN dot   = each UNIFORM probe's representative pixel (the 16px base grid).
//   • YELLOW dot  = each ADAPTIVE probe's representative pixel (the 8px refinement).
//
// Reads the flat atlas (screenProbePix, for repr-pixel positions + validity) + the indirection
// (tileHeader / tileIndices) exactly as the resolve does, so what you see is the actual probe set
// the cone pass resolves against. No tracing, no lighting — purely the placement map.

export const createDebugShaderMeta = (cfg: VoxelBakedConfig) => new ShaderMeta(
  {
    // Canvas dims (px) — the only live value. gw/gh (probe grid dims) are derived as
    // ceil(canvas / SP_TILE). (The tile + foveated ring thresholds are BAKED consts.)
    params: new VariableMeta("uParams", VariableKind.Uniform, `vec2<f32>`),
    // World normal G-buffer — used only for context (n.a<0.5 = no surface) + a dim shade.
    normalTex: new VariableMeta("normalTex", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "float",
    }),
    // Flat probe atlas: .xy = each probe's representative full-res pixel, .z = validity. rgba32float
    // → unfilterable-float, point-loaded (textureLoad) at the slot→texel address.
    screenProbePix: new VariableMeta("screenProbePix", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "unfilterable-float",
    }),
    // Adaptive indirection (group 1, StorageRead — read-only storage in the fragment stage): per
    // coarse tile, the count of adaptive probes + their global atlas slots (stride SP_K).
    tileHeader: new VariableMeta("uTileHeader", VariableKind.StorageRead, `array<u32>`, {
      visibility: GPUShaderStage.FRAGMENT,
    }),
    tileIndices: new VariableMeta("uTileIndices", VariableKind.StorageRead, `array<u32>`, {
      visibility: GPUShaderStage.FRAGMENT,
    }),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
// Max adaptive probes per coarse tile = the tileIndices stride (see voxelResources.SCREEN_PROBE_K).
const SP_K: u32 = ${SCREEN_PROBE_K}u;
// BAKED lattice pitch + foveated ring thresholds (the density boundaries drawn as cyan circles).
const SP_TILE: f32 = ${probeTile(cfg)};
const RING_R0: f32 = ${ringThresholds(cfg).r0};
const RING_R1: f32 = ${ringThresholds(cfg).r1};

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

// True when full-res pixel p is within a 3x3 dot of the probe pixel stored at atlas texel (chebyshev
// distance <= 1), and that probe is valid (.z > 0.5).
fn probe_dot(texel: vec2<i32>, p: vec2<i32>) -> bool {
  let rec = textureLoad(screenProbePix, texel, 0);
  if (rec.z < 0.5) { return false; }
  let pix = vec2<i32>(i32(rec.x), i32(rec.y));
  let d = abs(p - pix);
  return d.x <= 1 && d.y <= 1;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  let W = uParams.x;
  let H = uParams.y;
  let tile = i32(SP_TILE);
  let full = min(vec2<i32>(input.texCoord * vec2<f32>(W, H)), vec2<i32>(i32(W) - 1, i32(H) - 1));
  let gw = i32(ceil(W / SP_TILE));
  let gh = i32(ceil(H / SP_TILE));

  // Context: dim shade where the G-buffer has geometry, near-black elsewhere.
  let n = textureLoad(normalTex, full, 0);
  var col = select(vec3<f32>(0.02), vec3<f32>(0.10), n.a >= 0.5);

  // Faint coarse-tile grid.
  let tc = full / tile;
  let inTile = full - tc * tile;
  if (inTile.x == 0 || inTile.y == 0) { col = col + vec3<f32>(0.04); }

  // Foveated ring boundaries (r0 / r1) as faint cyan circles (same normalization as
  // sp_ring_base_level: screen-center distance, corner = 1).
  let rr = length(vec2<f32>(full) - vec2<f32>(W, H) * 0.5) / max(0.5 * length(vec2<f32>(W, H)), 1e-4);
  if (abs(rr - RING_R0) < 0.002 || abs(rr - RING_R1) < 0.002) {
    col = mix(col, vec3<f32>(0.0, 0.7, 0.9), 0.6);
  }

  if (tc.x >= 0 && tc.y >= 0 && tc.x < gw && tc.y < gh) {
    let tileIdx = tc.y * gw + tc.x;
    let cnt = min(SP_K, uTileHeader[u32(tileIdx)]);

    // Heat tint on subdivided tiles (more adaptive probes → hotter).
    if (cnt > 0u) {
      col = mix(col, vec3<f32>(0.7, 0.12, 0.0), 0.22 + 0.5 * f32(cnt) / f32(SP_K));
    }

    // UNIFORM / ring-block probes = GREEN dots. A probe's repr pixel lies anywhere within its
    // BLOCK (block center ± jitter — up to 3 tiles down-right of its ANCHOR tile at ring level 2),
    // so testing only this pixel's own tile texel missed every coarse-ring probe. Scan the anchors
    // whose block could contain this pixel (up to 3 back per axis); validity + the dot test reject
    // non-anchor texels. A REFINED tile's uniform probe stays EXCLUDED (tileHeader > 0 — the fine
    // lattice replaces it), so the view still shows the probe set the resolve actually uses.
    for (var ay = 0; ay < 4; ay = ay + 1) {
      for (var ax = 0; ax < 4; ax = ax + 1) {
        let t = tc - vec2<i32>(ax, ay);
        if (t.x < 0 || t.y < 0) { continue; }
        if (uTileHeader[u32(t.y * gw + t.x)] != 0u) { continue; }
        if (probe_dot(t, full)) { col = vec3<f32>(0.1, 1.0, 0.25); }
      }
    }

    // ADAPTIVE probes parented to this tile = YELLOW dots (drawn on top).
    for (var j = 0u; j < cnt; j = j + 1u) {
      let slot = i32(uTileIndices[u32(tileIdx) * SP_K + j]);
      if (probe_dot(vec2<i32>(slot % gw, slot / gw), full)) {
        col = vec3<f32>(1.0, 0.85, 0.0);
      }
    }
  }

  return vec4<f32>(col, 1.0);
}
`,
);
