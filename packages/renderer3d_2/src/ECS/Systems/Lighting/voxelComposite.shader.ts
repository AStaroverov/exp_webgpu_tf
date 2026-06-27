import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";

// VCT Layer 4 — the COMPOSITE: turn the indirect cone gather into the FINAL lit image.
//   final = albedo·(ambient·AO + directSun·shadow + indirect) + selfEmission.
// A fullscreen FULL-res pass over the G-buffer. Per pixel:
//   - albedo  = G-buffer albedo (the SDF renderTexture).
//   - directSun = the directional sun (N·L · color · intensity), with a CRISP cast shadow from
//     the sun-POV depth map (sun_shadow(): reconstruct world P, project into the sun's orthoZO
//     clip space, compare depth with normal-offset + slope bias + 5×5 tent PCF).
//   - indirect already carries giStrength (baked into the cone's rgb); AO = the cone's hemisphere
//     visibility (cone.a). The cone output is HALF-res → bilinear-upsampled here (linear sampler).
//     The emitters (point lights) live entirely in this indirect cone-GI term.
//   - self-emission makes emitters glow: read the per-pixel G-buffer emission target written by
//     fs_main (uColor.rgb·abs(material.x)). A SURFACE property → no voxel cross-contamination.

export const shaderMeta = new ShaderMeta(
  {
    // .x = ambient (the floor scaled by AO). giStrength is already baked into the cone's rgb,
    // so it is NOT re-applied here. .y = exposure (HDR multiplier before ACES tonemapping).
    // .z = sun shadow-map world texel size (world units per texel, for the normal-offset bias).
    // .w = penumbra softness strength — the PCF filter widens as the sun intensity (uSun.w) drops
    //      below 1, so a dimmer sun casts a softer, wider shadow edge (physical-ish penumbra).
    params: new VariableMeta("uParams", VariableKind.Uniform, `vec4<f32>`),
    // .x = screen width (px), .y = screen height (px) — for the half-res cone upsample uv. (z/w spare.)
    params2: new VariableMeta("uParams2", VariableKind.Uniform, `vec4<f32>`),
    // Directional sun: .xyz = normalized world dir TOWARD the sun, .w = effective intensity
    // (0 when disabled). A real parallel sun — its DIRECT term lights every surface by N·L.
    sun: new VariableMeta("uSun", VariableKind.Uniform, `vec4<f32>`),
    // .rgb = sun color (linear).
    sunColor: new VariableMeta("uSunColor", VariableKind.Uniform, `vec4<f32>`),
    // ── Sun shadow-map inputs (crisp directional cast shadow) ──────────────────────────
    // inverse(viewProj) (reverse-Z) to reconstruct world P from the camera depth.
    invViewProj: new VariableMeta("uInvViewProj", VariableKind.Uniform, `mat4x4<f32>`),
    // Sun orthographic view-projection (orthoZO, z in [0,1]) — projects P into the shadow map.
    sunViewProj: new VariableMeta("uSunViewProj", VariableKind.Uniform, `mat4x4<f32>`),
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
// compare depth. 1 = lit, 0 = shadowed. Normal-offset bias (uParams.z = world texel size) is the
// main acne killer; tiny slope-scaled constant bias on top. The map is orthoZO + clear 1.0 +
// "less-equal", so it stores the nearest-to-sun depth → shadowed when the fragment's sun-space
// depth is GREATER than stored (+ bias).
// Soft shadows: a per-pixel-rotated Poisson disk of radius = spread (texels). spread=1 is the
// near-crisp baseline; larger values widen the penumbra (driven by sun dimness in fs_main).
fn sun_shadow(P: vec3<f32>, N: vec3<f32>, ndl: f32, spread: f32, seed: vec2<f32>) -> f32 {
  let Po = P + N * (uParams.z * 2.5);
  let ls = uSunViewProj * vec4<f32>(Po, 1.0);
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

  // Indirect radiance (already ×giStrength) + hemisphere visibility (AO). coneTex is half-res;
  // the linear sampler bilinearly upsamples it. uParams2.xy = full dims.
  let cone = textureSampleLevel(coneTex, coneSampler, (vec2<f32>(pixel) + vec2<f32>(0.5)) / uParams2.xy, 0.0);
  let indirect = cone.rgb;
  let ao = cone.a;

  // Self-emission: per-pixel from the G-buffer emission target (surface property).
  let emission = textureLoad(emissionTex, pixel, 0).rgb;

  // Direct parallel sun (uSun.w = 0 when disabled), with a crisp shadow-map cast shadow.
  // N·L Lambert × sun color × intensity × shadow visibility. P reconstructed from camera depth.
  let ndl = max(dot(N, uSun.xyz), 0.0);
  var sunVis = 1.0;
  if (uSun.w > 0.0 && ndl > 0.0) {
    let depthP = textureLoad(depthTex, pixel, 0);
    let uvP = (vec2<f32>(pixel) + vec2<f32>(0.5)) / uParams2.xy;
    let P = unproject(vec3<f32>(uvP.x * 2.0 - 1.0, (1.0 - uvP.y) * 2.0 - 1.0, depthP));
    // Penumbra grows as the sun dims below 1: spread = 1 at full sun, up to 1 + penumbra at sun→0.
    let spread = 1.0 + uParams.w * clamp(1.0 - uSun.w, 0.0, 1.0);
    sunVis = sun_shadow(P, N, ndl, spread, input.position.xy);
  }
  let sunDirect = ndl * uSunColor.rgb * uSun.w * sunVis;

  let lit = albedo * (uParams.x * ao + sunDirect + indirect) + emission;
  // HDR → display: exposure (uParams.y) then ACES tonemap, so bright sources roll off instead of
  // clipping to flat white. (Gamma/sRGB encode is left to the present chain as today.)
  let mapped = aces(lit * uParams.y);
  return vec4f(mapped, 1.0);
}
`,
);
