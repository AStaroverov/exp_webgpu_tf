import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";
import { VoxelBakedConfig } from "./voxelConfig.ts";

// VCT Layer 4 — the COMPOSITE: turn the indirect cone gather into the FINAL lit image.
//   final = albedo·(ambient·AO + directSun·shadow + indirect) + selfEmission.
// A fullscreen FULL-res pass over the G-buffer. Per pixel:
//   - albedo  = G-buffer albedo (the SDF renderTexture).
//   - directSun = the directional sun (N·L · color · intensity), with a CRISP cast shadow from
//     the sun-POV depth map (sun_shadow(): reconstruct world P, project into the sun's orthoZO
//     clip space, compare depth with normal-offset + slope bias + 5×5 tent PCF).
//   - indirect already carries giStrength (baked into the cone's rgb); AO = the cone's hemisphere
//     visibility (cone.a). The cone output is HALF-res → normal-aware (bilateral) upsample here
//     (upsample_cone) so a near emitter's light does not smear across shape silhouettes.
//     The emitters (point lights) live entirely in this indirect cone-GI term.
//   - self-emission makes emitters glow: read the per-pixel G-buffer emission target written by
//     fs_main (uColor.rgb·abs(material.x)). A SURFACE property → no voxel cross-contamination.

export function createCompositeShaderMeta(cfg: VoxelBakedConfig) {
  return new ShaderMeta(
  {
    // All per-frame scalar/vector/matrix uniforms consolidated into ONE struct buffer (uF) so the
    // pass binds + uploads a single UBO instead of six. The WGSL `CompositeFrame` struct is defined
    // in the body below; the type name here is opaque to the meta system, so size/bufferSize are
    // given explicitly (48 f32 = 192 bytes: 4×vec4 + 2×mat4x4, all 16-byte aligned → no padding).
    // Fields:
    //   params  .z = sun shadow-map world texel size (normal-offset bias). .x/.y/.w baked consts.
    //   params2 .x/.y = screen width/height px (cone upsample uv), .z = cone downscale factor.
    //   sun     .xyz = normalized world dir TOWARD the sun, .w = effective intensity (0 = disabled).
    //   sunColor.rgb = sun color (linear).
    //   invViewProj = inverse(viewProj) (reverse-Z) → reconstruct world P from camera depth.
    //   sunViewProj = sun orthographic view-projection (orthoZO) → project P into the shadow map.
    frame: new VariableMeta("uF", VariableKind.Uniform, `CompositeFrame`, {
      size: 48,
      bufferSize: 192,
    }),
    // G-buffer reverse-Z camera depth, to reconstruct the per-pixel world position P.
    depthTex: new VariableMeta("depthTex", VariableKind.Texture, `texture_depth_2d`, {
      textureSampleType: "depth",
    }),
    // Sun-POV depth map (depth32float from the sunDepth pass). textureLoad (no sampler).
    shadowMap: new VariableMeta("shadowMap", VariableKind.Texture, `texture_depth_2d`, {
      textureSampleType: "depth",
    }),
    // G-buffer albedo (the SDF draw-pass renderTexture).
    albedoTex: new VariableMeta("albedoTex", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "float",
    }),
    // G-buffer world normal (rgba16float, packed *0.5+0.5; a<0.5 = no surface) — used only as the
    // surface mask here.
    normalTex: new VariableMeta("normalTex", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "float",
    }),
    // Cone output: rgb = indirect (×giStrength), a = AO visibility. HALF-res → sampled with
    // a linear sampler to bilinear-upsample to full res.
    coneTex: new VariableMeta("coneTex", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "float",
    }),
    // Linear/clamp sampler for the bilinear upsample of the half-res cone output.
    coneSampler: new VariableMeta("coneSampler", VariableKind.Sampler, `sampler`),
    // G-buffer per-pixel self-emission (rgba16float, rgb = uColor·abs(material.x)).
    emissionTex: new VariableMeta("emissionTex", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "float",
    }),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
const AMBIENT: f32 = ${cfg.ambient};
const EXPOSURE: f32 = ${cfg.exposure};
const PENUMBRA: f32 = ${cfg.penumbra};
const SHADOW_BASE_SPREAD: f32 = ${cfg.shadowBaseSpread};

// Per-frame uniforms, one consolidated UBO (uF). All members are 16-byte aligned (vec4 / mat4x4)
// so the std140 layout is dense: params@0, params2@16, sun@32, sunColor@48, invViewProj@64,
// sunViewProj@128 (bytes). Mirrored by the CPU scratch layout in createVoxelSystem.composite().
struct CompositeFrame {
  params: vec4<f32>,
  params2: vec4<f32>,
  sun: vec4<f32>,
  sunColor: vec4<f32>,
  invViewProj: mat4x4<f32>,
  sunViewProj: mat4x4<f32>,
};

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
  let w = uF.invViewProj * vec4<f32>(ndc, 1.0);
  return w.xyz / w.w;
}

// 16-sample Poisson disk (unit disk). A scattered, low-discrepancy tap set: scaling it gives a
// wide soft kernel WITHOUT the regular-grid banding a square 5×5 kernel produces at large radii.
const POISSON16 = array<vec2<f32>, 16>(
  vec2<f32>(-0.94201624, -0.39906216), vec2<f32>( 0.94558609, -0.76890725),
  vec2<f32>(-0.09418410, -0.92938870), vec2<f32>( 0.34495938,  0.29387760),
  vec2<f32>(-0.91588581,  0.45771432), vec2<f32>(-0.81544232, -0.87912464),
  vec2<f32>(-0.38277543,  0.27676845), vec2<f32>( 0.97484398,  0.75648379),
  vec2<f32>( 0.44323325, -0.97511554), vec2<f32>( 0.53742981, -0.47373420),
  vec2<f32>(-0.26496911, -0.41893023), vec2<f32>( 0.79197514,  0.19090188),
  vec2<f32>(-0.24188840,  0.99706507), vec2<f32>(-0.81409955,  0.91437590),
  vec2<f32>( 0.19984126,  0.78641367), vec2<f32>( 0.14383161, -0.14100790)
);

// Interleaved gradient noise (Jimenez) — a cheap per-pixel pseudo-random scalar in [0,1). Used to
// rotate the Poisson disk per fragment so the residual under-sampling shows up as fine noise the
// eye reads as softness, instead of correlated stair-steps.
fn ign(p: vec2<f32>) -> f32 {
  return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

// Sun shadow-map lookup: project the surface point P into the sun's orthographic clip space and
// compare depth. 1 = lit, 0 = shadowed. Normal-offset bias (uF.params.z = world texel size) is the
// main acne killer; tiny slope-scaled constant bias on top. The map is orthoZO + clear 1.0 +
// "less-equal", so it stores the nearest-to-sun depth → shadowed when the fragment's sun-space
// depth is GREATER than stored (+ bias).
// Soft shadows: a per-pixel-rotated Poisson disk of radius = spread (texels). spread=1 is the
// near-crisp baseline; larger values widen the penumbra (driven by sun dimness in fs_main).
fn sun_shadow(P: vec3<f32>, N: vec3<f32>, ndl: f32, spread: f32, seed: vec2<f32>) -> f32 {
  let Po = P + N * (uF.params.z * 2.5);
  let ls = uF.sunViewProj * vec4<f32>(Po, 1.0);
  let ndc = ls.xyz / ls.w;
  var uv = ndc.xy * 0.5 + vec2<f32>(0.5, 0.5);
  uv.y = 1.0 - uv.y;
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
    return 1.0;
  }
  let bias = 0.0004 + 0.0015 * (1.0 - ndl);
  let dim = vec2<i32>(textureDimensions(shadowMap, 0));
  let texelPos = uv * vec2<f32>(dim) - vec2<f32>(0.5);
  // Per-pixel rotation of the whole disk → decorrelates the taps so banding becomes noise.
  let ang = ign(seed) * 6.2831853;
  let ca = cos(ang);
  let sa = sin(ang);
  var sum = 0.0;
  for (var i = 0; i < 16; i = i + 1) {
    let o = POISSON16[i];
    let r = vec2<f32>(o.x * ca - o.y * sa, o.x * sa + o.y * ca) * spread;
    let c = clamp(vec2<i32>(round(texelPos + r)), vec2<i32>(0, 0), dim - vec2<i32>(1, 1));
    let s = textureLoad(shadowMap, c, 0);
    sum = sum + select(0.0, 1.0, ndc.z <= s + bias);
  }
  return sum / 16.0;
}

// ACES filmic tonemap (Narkowicz approximation): compresses unbounded HDR into [0,1] with a
// highlight roll-off, so bright emitters/sun keep their shape instead of clipping to flat white
// (and a sun-shadow under a bright light reads as a soft dip, not a hard black step on white).
fn aces(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Normal-aware (bilateral) upsample of the HALF-res cone output. Plain bilinear bleeds a half-res
// texel's value ~2 full-res px past a silhouette, so when a bright emitter passes near a small shape
// its lit value smears onto the shape's edge (a blown rim, worse when zoomed out — the edge is a
// bigger fraction of the shape). Fix: blend the 4 nearest half-res taps but weight each by how well
// its surface normal matches THIS pixel's normal, and drop taps that sit on background (a<0.5). Taps
// across a silhouette (different orientation, or no surface) stop contributing → the edge stays crisp.
// The +1e-4 floor degrades gracefully to plain bilinear if every tap is rejected (never worse than
// today). No derivatives are used, so this is safe in the post-early-return (non-uniform) flow.
fn upsample_cone(pixel: vec2<i32>, cn: vec3<f32>) -> vec4<f32> {
  let fullDim = vec2<i32>(textureDimensions(depthTex, 0));
  let coneDim = vec2<i32>(textureDimensions(coneTex, 0));
  // Cone downscale factor (2 = half-res, 4 = quarter-res), set on the CPU.
  let s = max(1.0, uF.params2.z);
  let si = i32(s);
  // Center pixel mapped into cone-res texel space (texel centers at integer+0.5).
  let hp = (vec2<f32>(pixel) + vec2<f32>(0.5)) / s - vec2<f32>(0.5, 0.5);
  let base = vec2<i32>(floor(hp));
  let fr = hp - floor(hp);
  var sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var wsum = 0.0;
  for (var oy = 0; oy <= 1; oy = oy + 1) {
    for (var ox = 0; ox <= 1; ox = ox + 1) {
      let h = clamp(base + vec2<i32>(ox, oy), vec2<i32>(0, 0), coneDim - vec2<i32>(1, 1));
      let bw = select(1.0 - fr.x, fr.x, ox == 1) * select(1.0 - fr.y, fr.y, oy == 1);
      // Representative full-res sample for this cone texel (center of the s×s block it covers).
      let fp = clamp(h * si + vec2<i32>(si / 2, si / 2), vec2<i32>(0, 0), fullDim - vec2<i32>(1, 1));
      let nt = textureLoad(normalTex, fp, 0);
      let ntDir = normalize(nt.rgb * 2.0 - 1.0);
      // Background taps (a<0.5) contribute 0 so masked regions never bleed onto a silhouette.
      let nwgt = select(0.0, pow(max(dot(ntDir, cn), 0.0), 4.0), nt.a >= 0.5);
      let w = bw * (nwgt + 1e-4);
      sum = sum + textureLoad(coneTex, h, 0) * w;
      wsum = wsum + w;
    }
  }
  return sum / wsum;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  let pixel = vec2<i32>(floor(input.position.xy));

  // World normal G-buffer; a<0.5 = no surface (background) → leave black.
  let n = textureLoad(normalTex, pixel, 0);
  if (n.a < 0.5) {
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }
  let N = normalize(n.rgb * 2.0 - 1.0);

  let albedo = textureLoad(albedoTex, pixel, 0).rgb;

  // Indirect radiance (already ×giStrength) + hemisphere visibility (AO). coneTex is half-res →
  // normal-aware bilateral upsample (keeps emitter light from smearing onto shape silhouettes).
  let cone = upsample_cone(pixel, N);
  let indirect = cone.rgb;
  let ao = cone.a;

  // Self-emission: per-pixel from the G-buffer emission target (surface property).
  let emission = textureLoad(emissionTex, pixel, 0).rgb;

  // Direct parallel sun (uF.sun.w = 0 when disabled), with a crisp shadow-map cast shadow.
  // N·L Lambert × sun color × intensity × shadow visibility. P reconstructed from camera depth.
  let ndl = max(dot(N, uF.sun.xyz), 0.0);
  var sunVis = 1.0;
  if (uF.sun.w > 0.0 && ndl > 0.0) {
    let depthP = textureLoad(depthTex, pixel, 0);
    let uvP = (vec2<f32>(pixel) + vec2<f32>(0.5)) / uF.params2.xy;
    let P = unproject(vec3<f32>(uvP.x * 2.0 - 1.0, (1.0 - uvP.y) * 2.0 - 1.0, depthP));
    // Base PCF softness ALWAYS applied (kills the shadow-map texel staircase even at full sun),
    // and grows further as the sun dims below 1 (a dimmer sun → softer, wider penumbra).
    let spread = SHADOW_BASE_SPREAD + PENUMBRA * clamp(1.0 - uF.sun.w, 0.0, 1.0);
    sunVis = sun_shadow(P, N, ndl, spread, input.position.xy);
  }
  let sunDirect = ndl * uF.sunColor.rgb * uF.sun.w * sunVis;

  let lit = albedo * (AMBIENT * ao + sunDirect + indirect) + emission;
  // HDR → display: exposure (EXPOSURE) then ACES tonemap, so bright sources roll off instead of
  // clipping to flat white. (Gamma/sRGB encode is left to the present chain as today.)
  let mapped = aces(lit * EXPOSURE);
  return vec4f(mapped, 1.0);
}
`,
  );
}
