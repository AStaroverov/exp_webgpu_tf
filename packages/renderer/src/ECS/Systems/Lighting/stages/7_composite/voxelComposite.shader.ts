import { VariableKind, VariableMeta } from "../../../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../../../WGSL/wgsl.ts";
import { VoxelBakedConfig } from "../../core/voxelConfig.ts";

// VCT Layer 4 — the COMPOSITE: turn the indirect cone gather into the FINAL lit image.
//   final = albedo·(ambient·AO + directSun·shadow + indirect) + selfEmission.
// A fullscreen FULL-res pass over the G-buffer. Per pixel:
//   - albedo  = G-buffer albedo (the SDF renderTexture).
//   - directSun = the directional sun (N·L · color · intensity), with a DF-style cast shadow:
//     the cone pass traced one cone per half-res pixel toward the sun through the voxel field
//     (penumbra grows with occluder distance) — this pass just upsamples its sunVis target.
//   - indirect already carries giStrength (baked into the cone's rgb); AO = the cone's hemisphere
//     visibility (cone.a). The cone output is HALF-res → normal-aware (bilateral) upsample here
//     (upsample_cone) so a near emitter's light does not smear across shape silhouettes.
//     The emitters (point lights) live entirely in this indirect cone-GI term.
//   - self-emission makes emitters glow: read the per-pixel G-buffer emission target written by
//     fs_main (uColor.rgb·abs(material.x)). A SURFACE property → no voxel cross-contamination.

export function createCompositeShaderMeta(cfg: VoxelBakedConfig) {
  return new ShaderMeta(
    {
      // All per-frame scalar/vector uniforms consolidated into ONE struct buffer (uF) so the pass
      // binds + uploads a single UBO. The WGSL `CompositeFrame` struct is defined in the body
      // below; the type name is opaque to the meta system, so size/bufferSize are explicit
      // (12 f32 = 48 bytes: 3×vec4, 16-byte aligned → no padding). Fields:
      //   params2 .x/.y = screen width/height px (cone upsample uv), .z = cone downscale factor.
      //   sun     .xyz = normalized world dir TOWARD the sun, .w = effective intensity (0 = disabled).
      //   sunColor.rgb = sun color (linear).
      frame: new VariableMeta("uF", VariableKind.Uniform, `CompositeFrame`, {
        size: 12,
        bufferSize: 48,
      }),
      // G-buffer reverse-Z camera depth (full-res dims for the cone upsample mapping).
      depthTex: new VariableMeta("depthTex", VariableKind.Texture, `texture_depth_2d`, {
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
      // SUN cast-shadow visibility from the cone pass (r16float, half-res, @location(1)) — the
      // only sun-shadow source. Bilinear-upsampled via coneSampler.
      sunVisTex: new VariableMeta("sunVisTex", VariableKind.Texture, `texture_2d<f32>`, {
        textureSampleType: "float",
      }),
    },
    {},
    // language=WGSL
    wgsl /* wgsl */ `
const AMBIENT: f32 = ${cfg.ambient};
const EXPOSURE: f32 = ${cfg.exposure};

// Per-frame uniforms, one consolidated UBO (uF): params2@0, sun@16, sunColor@32 (bytes).
// Mirrored by the CPU scratch layout in compositeSystem.composite().
struct CompositeFrame {
  params2: vec4<f32>,
  sun: vec4<f32>,
  sunColor: vec4<f32>,
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
    // The cone pass traced one cone toward the sun per half-res pixel — bilinear-upsample its
    // visibility target. Penumbra width came from the trace (grows with occluder distance).
    let uvP = (vec2<f32>(pixel) + vec2<f32>(0.5)) / uF.params2.xy;
    sunVis = textureSampleLevel(sunVisTex, coneSampler, uvP, 0.0).r;
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
