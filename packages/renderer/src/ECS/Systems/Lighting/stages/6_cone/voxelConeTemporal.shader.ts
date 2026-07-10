import { VariableKind, VariableMeta } from "../../../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../../../WGSL/wgsl.ts";
import { unprojectWGSL } from "../../core/shaders/voxelTrace.wgsl.ts";
import type { VoxelBakedConfig } from "../../core/voxelConfig.ts";

// CONE-OUTPUT TEMPORAL FILTER ("point C") — a fullscreen pass over the HALF-res cone resolve
// output, the last line of defense against the whole class of RESOLVE noise the probe-atlas
// temporal cannot reach (it stabilizes each probe's SH — the INPUT of the resolve — but not what
// the resolve does with it):
//   - probe-SET churn: a tile entering/leaving refinement, a foveated block flipping level, an
//     adaptive probe (de)spawning — kernel weights jump while every probe is individually stable;
//   - the per-pixel AO cones (jittered every frame, no history of their own);
//   - the residual motion noise after the anchor-distance gate dumps probe history during zoom/pan
//     (the SURFACE reprojection here stays valid exactly when the PROBE reprojection is not).
//
// Per half-res pixel: reconstruct the world position P from the full-res reverse-Z depth (same
// texCoord mapping the cone pass uses), reproject P into LAST frame's clip (uPrevViewProj,
// forward), point-load the NEAREST history texel (last frame's FILTERED output), CLAMP it to the
// plus-shaped (5-tap) neighborhood min/max of the CURRENT raw frame (the classic TAA anti-ghosting
// clamp — a real lighting change punches through immediately, only noise gets averaged; this is
// what keeps the probe layer's "lagging corona" bug from reappearing here), and blend:
// out = mix(raw, hist, h).
// The AO channel (.a) rides the same vec4 clamp+blend. h = 0 → exact passthrough (the A/B).
//
// No sub-pixel jitter exists anywhere in this renderer, so this is pure ACCUMULATION (denoising),
// not TAA-style reconstruction — it cannot soften G-buffer edges (the composite's normal-aware
// upsample stays the edge authority); it only smooths the low-frequency GI/AO field.
//
// The CPU side (coneSystem) runs: cone → coneOutput (raw) → THIS pass → coneFiltered (what the
// composite samples) → copyTextureToTexture(coneFiltered → coneHistory). The copy (a half-res
// rgba16float, ~2 MB) buys a parity-free design: the composite always samples ONE texture and no
// consumer needs ping-pong bind-group variants.

export const createConeTemporalShaderMeta = (cfg: VoxelBakedConfig) => new ShaderMeta(
  {
    // .x = canvas width (px), .y = canvas height (px) — FULL-res dims (the depth lookup + world
    // reconstruction run in full-res pixel space; the pass itself renders at the cone target's
    // half res, mapped via texCoord). .z = history VALIDITY (0 or 1): the hysteresis itself is the
    // BAKED CONE_HYSTERESIS; this lane only zeroes it for one frame after a resize/scale recreate
    // while the history texture is zero-filled. .w spare.
    params: new VariableMeta("uParams", VariableKind.Uniform, `vec4<f32>`),
    // inverse(viewProjMatrix) (reverse-Z) — world-position reconstruction (this frame).
    invViewProj: new VariableMeta("uInvViewProj", VariableKind.Uniform, `mat4x4<f32>`),
    // LAST frame's FORWARD viewProj — reprojects the fresh world anchor into the previous frame.
    prevViewProj: new VariableMeta("uPrevViewProj", VariableKind.Uniform, `mat4x4<f32>`),
    // Full-res reverse-Z G-buffer depth (the same source the cone pass reconstructs P from).
    depthTex: new VariableMeta("depthTex", VariableKind.Texture, `texture_depth_2d`, {
      textureSampleType: "depth",
    }),
    // THIS frame's raw cone-resolve output (rgb = indirect, a = AO) — point-loaded.
    coneRaw: new VariableMeta("coneRaw", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "float",
    }),
    // LAST frame's filtered output (the history) — point-loaded at the NEAREST reprojected texel.
    // Bilinear was tried and REVERTED: the history has no bilateral weight, so filtering mixed
    // texels ACROSS silhouettes and leaked emitter light onto adjacent surfaces as a bright rim.
    coneHist: new VariableMeta("coneHist", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "float",
    }),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
${unprojectWGSL}

// BAKED history weight (0 = passthrough; the neighborhood clamp keeps high values honest — a real
// lighting change punches through in 1 frame). Rebuild to change.
const CONE_HYSTERESIS: f32 = ${cfg.coneTemporalHysteresis};

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

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  let h = CONE_HYSTERESIS * uParams.z; // × history-validity (0 for one frame after a recreate)
  // This pass renders at the cone target's res — position.xy IS the half-res pixel.
  let half = vec2<i32>(floor(input.position.xy));
  let raw = textureLoad(coneRaw, half, 0);
  if (h <= 0.0) { return raw; }

  // Representative full-res pixel (same texCoord mapping as the cone pass) → reverse-Z depth.
  // Cleared depth (0 = far plane) ⇒ sky / no surface ⇒ nothing to reproject.
  let full = min(vec2<i32>(input.texCoord * uParams.xy), vec2<i32>(uParams.xy) - vec2<i32>(1));
  let depth = textureLoad(depthTex, full, 0);
  if (depth <= 0.0) { return raw; }

  // World position this frame → LAST frame's uv (the exact uv↔ndc mapping the probe gather uses,
  // including the Y flip). Off-frustum / behind the prev camera ⇒ fresh only.
  let uv = (vec2<f32>(full) + vec2<f32>(0.5)) / uParams.xy;
  let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth);
  let P = unproject(ndc, uInvViewProj);
  let prevClip = uPrevViewProj * vec4<f32>(P, 1.0);
  if (prevClip.w <= 1e-4) { return raw; }
  let prevNdc = prevClip.xyz / prevClip.w;
  let prevUv = vec2<f32>(prevNdc.x * 0.5 + 0.5, 0.5 - prevNdc.y * 0.5);
  if (any(prevUv < vec2<f32>(0.0)) || any(prevUv >= vec2<f32>(1.0))
      || prevNdc.z < 0.0 || prevNdc.z > 1.0) {
    return raw;
  }

  // NEAREST history texel (a static camera reprojects onto its OWN texel — exact, no cross-edge
  // mixing; motion is off by ≤0.5 texel, bounded by the clamp — fine for low-frequency GI).
  let dims = vec2<i32>(textureDimensions(coneRaw));
  let ht = clamp(vec2<i32>(prevUv * vec2<f32>(dims)), vec2<i32>(0), dims - vec2<i32>(1));
  // PLUS-shaped (5-tap) neighborhood min/max of the CURRENT raw frame — the anti-ghosting clamp.
  // History outside this box is a real change (an emitter moved, a shadow swept past) and gets
  // clamped onto it immediately; history inside it is temporal noise and averages out. Doubles as
  // the disocclusion guard, so no history depth/normal set is needed. The plus (not 3×3) keeps the
  // box tight at silhouettes: the diagonal corners let the OTHER side of an edge into the box, and
  // leaked bright values then survived the clamp as a persistent rim.
  var mn = raw;
  var mx = raw;
  for (var i = 0; i < 4; i = i + 1) {
    var o = vec2<i32>(0);
    if (i == 0) { o = vec2<i32>(1, 0); }
    else if (i == 1) { o = vec2<i32>(-1, 0); }
    else if (i == 2) { o = vec2<i32>(0, 1); }
    else { o = vec2<i32>(0, -1); }
    let s = textureLoad(coneRaw, clamp(half + o, vec2<i32>(0), dims - vec2<i32>(1)), 0);
    mn = min(mn, s);
    mx = max(mx, s);
  }
  let hist = clamp(textureLoad(coneHist, ht, 0), mn, mx);
  return mix(raw, hist, h);
}
`,
);
