import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";

// VCT Layer 4 — the COMPOSITE: turn the indirect cone gather into the FINAL lit image.
//   final = albedo·(ambient·AO + indirect) + selfEmission.
// A fullscreen FULL-res pass over the G-buffer. Per pixel:
//   - albedo  = G-buffer albedo (the SDF renderTexture).
//   - indirect already carries giStrength (baked into the cone's rgb); AO = the cone's
//     hemisphere visibility (cone.a). The cone output is HALF-res, so it is bilinear-
//     upsampled here (linear sampler) — indirect light is low-frequency, so that is fine.
//   - self-emission makes emitters glow: read the per-pixel G-buffer emission target written
//     by fs_main (uColor.rgb·abs(material.x)). It is a SURFACE property, so there is no voxel
//     cross-contamination / flicker as instances intersect.
// UNIFIED lighting model: ALL light (the sun — now a regular emitter — and every other emitter)
// arrives through the cone-GI 'indirect' term, which gathers the voxel radiance volume. There is
// NO separate sun direct term and NO per-emitter direct term here (those were the removed B-lite
// path); shadows/soft-shadows are the cone gather's per-direction occupancy.

export const shaderMeta = new ShaderMeta(
  {
    // .x = ambient (the floor scaled by AO). giStrength is already baked into the cone's rgb,
    // so it is NOT re-applied here. (y/z/w spare.)
    params: new VariableMeta("uParams", VariableKind.Uniform, `vec4<f32>`),
    // .x = screen width (px), .y = screen height (px) — for the half-res cone upsample uv. (z/w spare.)
    params2: new VariableMeta("uParams2", VariableKind.Uniform, `vec4<f32>`),
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

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  let pixel = vec2<i32>(floor(input.position.xy));

  // World normal G-buffer; a<0.5 = no surface (background) → leave black.
  let n = textureLoad(normalTex, pixel, 0);
  if (n.a < 0.5) {
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }

  let albedo = textureLoad(albedoTex, pixel, 0).rgb;

  // Indirect radiance (already ×giStrength) + hemisphere visibility (AO). coneTex is half-res;
  // the linear sampler bilinearly upsamples it. uParams2.xy = full dims.
  let cone = textureSampleLevel(coneTex, coneSampler, (vec2<f32>(pixel) + vec2<f32>(0.5)) / uParams2.xy, 0.0);
  let indirect = cone.rgb;
  let ao = cone.a;

  // Self-emission: per-pixel from the G-buffer emission target (surface property).
  let emission = textureLoad(emissionTex, pixel, 0).rgb;

  let lit = albedo * (uParams.x * ao + indirect) + emission;
  return vec4f(lit, 1.0);
}
`,
);
