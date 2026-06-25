import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";

// VCT Layer 4 — the COMPOSITE: turn the indirect cone gather into the FINAL lit image.
//   final = albedo·(ambient·AO + directSun + indirect) + selfEmission.
// A fullscreen pass over the G-buffer (full-res, so NO bilateral upsample is needed — that
// is a later perf option). Per pixel:
//   - albedo  = G-buffer albedo (the SDF renderTexture).
//   - indirect already carries giStrength (baked into the cone's rgb); AO = the cone's
//     hemisphere visibility (cone.a).
//   - direct sun is UNSHADOWED in v1 (ndl·sunColor·intensity); contact occlusion comes from
//     the indirect/AO terms, not a shadow ray.
//   - self-emission makes emitters glow: read the per-pixel G-buffer emission target written
//     by fs_main (uColor.rgb·abs(material.x)). It is a SURFACE property, so there is no
//     voxel cross-contamination and no flicker as instances intersect (the previous
//     voxelEmission nearest-instance read did both).
// Fullscreen pass over the G-buffer; textureLoad (integer coords) for every read, no sampler.

export const shaderMeta = new ShaderMeta(
  {
    // .x = ambient (the floor scaled by AO). z/w spare; giStrength is already baked into
    // the cone's rgb, so it is NOT re-applied here.
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
    // Cone output: rgb = indirect (×giStrength), a = AO visibility.
    coneTex: new VariableMeta("coneTex", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "float",
    }),
    // G-buffer per-pixel self-emission (rgba16float, rgb = uColor·abs(material.x)).
    // A surface property — replaces the old voxelEmission nearest-instance read.
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

  // Indirect radiance (already ×giStrength) + hemisphere visibility (AO).
  let cone = textureLoad(coneTex, pixel, 0);
  let indirect = cone.rgb;
  let ao = cone.a;

  // Direct sun, unshadowed in v1: contact occlusion comes from indirect/AO instead.
  let L = uSun.xyz;
  let ndl = max(dot(N, L), 0.0);
  let direct = ndl * uSunColor.rgb * uSun.w;

  // Self-emission: per-pixel from the G-buffer emission target (surface property).
  // No voxel cross-contamination / flicker — emitters HDR-glow correctly.
  let emission = textureLoad(emissionTex, pixel, 0).rgb;

  let lit = albedo * (uParams.x * ao + direct + indirect) + emission;
  return vec4f(lit, 1.0);
}
`,
);
