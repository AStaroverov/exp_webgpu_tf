import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";

// VCT Layer 4 — the COMPOSITE: turn the indirect cone gather into the FINAL lit image.
//   final = albedo·(ambient·AO + directSun + indirect) + selfEmission.
// A fullscreen FULL-res pass over the G-buffer. Per pixel:
//   - albedo  = G-buffer albedo (the SDF renderTexture).
//   - indirect already carries giStrength (baked into the cone's rgb); AO = the cone's
//     hemisphere visibility (cone.a). The cone output is HALF-res, so it is bilinear-
//     upsampled here (linear sampler) — indirect light is low-frequency, so that is fine.
//   - direct sun is UNSHADOWED (ndl·sunColor·intensity); contact occlusion + soft shadows come
//     from the indirect/AO terms (the cone gather's per-direction occupancy), not a shadow ray.
//   - self-emission makes emitters glow: read the per-pixel G-buffer emission target written
//     by fs_main (uColor.rgb·abs(material.x)). It is a SURFACE property, so there is no
//     voxel cross-contamination and no flicker as instances intersect (the previous
//     voxelEmission nearest-instance read did both).
// Fullscreen pass over the G-buffer; textureLoad (integer coords) for the G-buffer reads,
// textureSampleLevel (explicit LOD, uniform-control-flow safe) for the half-res cone upsample.

export const shaderMeta = new ShaderMeta(
  {
    // .x = ambient (the floor scaled by AO). giStrength is already baked into the cone's rgb,
    // so it is NOT re-applied here. .y = sun shadow-map enable (1/0). .z = sun shadow-map world
    // texel size (world units per texel, for the normal-offset bias). (w spare.)
    params: new VariableMeta("uParams", VariableKind.Uniform, `vec4<f32>`),
    // .x = screen width (px), .y = screen height (px). (z/w spare.)
    params2: new VariableMeta("uParams2", VariableKind.Uniform, `vec4<f32>`),
    // Directional sun: .xyz = normalized world dir TOWARD sun, .w = effective intensity
    // (0 when the sun is disabled) — same packing as voxelize's uSun.
    sun: new VariableMeta("uSun", VariableKind.Uniform, `vec4<f32>`),
    // .rgb = sun color (linear).
    sunColor: new VariableMeta("uSunColor", VariableKind.Uniform, `vec4<f32>`),
    // G-buffer albedo (the SDF draw-pass renderTexture).
    albedoTex: new VariableMeta("albedoTex", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "float",
    }),
    // G-buffer world normal (rgba16float, packed *0.5+0.5; a<0.5 = no surface).
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
    // A surface property — replaces the old voxelEmission nearest-instance read.
    emissionTex: new VariableMeta("emissionTex", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "float",
    }),
    // ── Sun shadow-map inputs ─────────────────────────────────────────────────────────
    // inverse(viewProj) (reverse-Z) to reconstruct world P from the camera depth (same as
    // the cone pass).
    invViewProj: new VariableMeta("uInvViewProj", VariableKind.Uniform, `mat4x4<f32>`),
    // Sun orthographic view-projection (orthoZO, z in [0,1]) — projects the reconstructed
    // world P into the shadow map. SAME matrix uploaded to sunShadow.shader's uViewProj.
    sunViewProj: new VariableMeta("uSunViewProj", VariableKind.Uniform, `mat4x4<f32>`),
    // G-buffer reverse-Z depth, to reconstruct the per-pixel world position P.
    depthTex: new VariableMeta("depthTex", VariableKind.Texture, `texture_depth_2d`, {
      textureSampleType: "depth",
    }),
    // Sun shadow map (depth32float from the sunShadow depth pass). Read via textureLoad
    // (integer coords, no sampler — a depth texture cannot use a filtering sampler, and a
    // non-filtering raw-depth read is exactly what the manual 2x2 PCF wants).
    shadowMap: new VariableMeta("shadowMap", VariableKind.Texture, `texture_depth_2d`, {
      textureSampleType: "depth",
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

// Sun shadow-map lookup: project the surface point P into the sun's orthographic clip space,
// sample the shadow map (nearest sun-facing depth per texel) and compare. Returns 1 = lit,
// 0 = shadowed (with 2x2 PCF averaging). The map is rendered by the sunShadow depth pass
// (orthoZO + depthClearValue 1.0 + depthCompare "less-equal"), so it stores the SMALLEST
// (nearest-to-sun) depth per texel: a fragment is shadowed when its own sun-space depth is
// GREATER than the stored depth (+ bias). Crisp SDF silhouettes (vs the old voxel march).
fn sun_shadow(P: vec3<f32>, N: vec3<f32>, ndl: f32) -> f32 {
  // NORMAL-OFFSET bias: push the receiver off its surface along the normal by a few shadow
  // texels (world units = uParams.z). This is the primary acne killer — it scales with the
  // map's world resolution and barely peter-pans, so the constant depth bias stays tiny.
  let texelW = uParams.z;
  let Po = P + N * (texelW * 2.5);

  let ls = uSunViewProj * vec4<f32>(Po, 1.0);
  let ndc = ls.xyz / ls.w;                 // x,y in [-1,1]; z in [0,1] (orthoZO)
  // NDC → texture UV. WebGPU texture origin is top-left and clip-space is Y-up → flip V.
  var uv = ndc.xy * 0.5 + vec2<f32>(0.5, 0.5);
  uv.y = 1.0 - uv.y;
  // Outside the sun frustum (or past near/far) → no shadow data → treat as LIT.
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 ||
      ndc.z < 0.0 || ndc.z > 1.0) {
    return 1.0;
  }

  // Slope-scaled constant bias: tiny base + more at grazing angles (where one texel spans
  // a large depth range). With the normal offset above, the base can stay very small.
  let bias = 0.0004 + 0.0015 * (1.0 - ndl);

  // TENT-WEIGHTED PCF over a (2R+1)^2 kernel about the bilinear sub-texel center → the edge
  // becomes a smooth gradient instead of hard stair-steps. textureLoad (integer coords — a
  // depth texture takes no filtering sampler); clamp taps to the map bounds. Standard depth:
  // a tap is LIT when the fragment depth <= stored (nearest the sun saw) + bias.
  let dim = vec2<i32>(textureDimensions(shadowMap, 0));
  let texelPos = uv * vec2<f32>(dim) - vec2<f32>(0.5);
  let base = vec2<i32>(floor(texelPos));
  let f = texelPos - floor(texelPos);
  let R = 2;
  let denom = f32(R) + 1.0;
  var sum = 0.0;
  var wsum = 0.0;
  for (var oy = -R; oy <= R; oy = oy + 1) {
    for (var ox = -R; ox <= R; ox = ox + 1) {
      let c = clamp(base + vec2<i32>(ox, oy), vec2<i32>(0, 0), dim - vec2<i32>(1, 1));
      let s = textureLoad(shadowMap, c, 0);
      let litTap = select(0.0, 1.0, ndc.z <= s + bias);
      let wx = max(0.0, 1.0 - abs(f32(ox) - f.x) / denom);
      let wy = max(0.0, 1.0 - abs(f32(oy) - f.y) / denom);
      let w = wx * wy;
      sum = sum + litTap * w;
      wsum = wsum + w;
    }
  }
  return sum / wsum;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  let pixel = vec2<i32>(floor(input.position.xy));

  // World normal from the G-buffer; a<0.5 = no surface (background) → leave black.
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

  // Direct sun, now with a sun shadow-map lookup (uParams.y enables it). The contact
  // occlusion + emitter penumbra still come from the indirect/AO terms; this adds the sun's
  // own crisp cast shadow to the final image. Reconstruct P from the camera reverse-Z depth
  // only when the shadow is needed, then re-project P with the sun's standard-depth matrix.
  let L = uSun.xyz;
  let ndl = max(dot(N, L), 0.0);
  var sunVis = 1.0;
  if (uParams.y > 0.5 && uSun.w > 0.0 && ndl > 0.0) {
    let depth = textureLoad(depthTex, pixel, 0);
    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) / uParams2.xy;
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth);
    let P = unproject(ndc);
    sunVis = sun_shadow(P, N, ndl);
  }
  let direct = ndl * uSunColor.rgb * uSun.w * sunVis;

  // Self-emission: per-pixel from the G-buffer emission target (surface property).
  // No voxel cross-contamination / flicker — emitters HDR-glow correctly.
  let emission = textureLoad(emissionTex, pixel, 0).rgb;

  let lit = albedo * (uParams.x * ao + direct + indirect) + emission;
  return vec4f(lit, 1.0);
}
`,
);
