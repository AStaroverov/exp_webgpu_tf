import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";

// Stage 2.3c — edge-aware blur of the low-res probe irradiance, to kill the visible probe
// grid before the resolve upsamples it. A 5×5 Gaussian at PROBE resolution (cheap), with
// per-tap weights rejecting probes on a different surface (normal + planar-depth), so it
// smooths within surfaces but not across edges. blurSigma≈0 → passthrough.

export const shaderMeta = new ShaderMeta(
  {
    // .x = probe spacing (px), .y = normalSharpness, .z = planeSigma, .w = blurSigma (probes).
    params: new VariableMeta("uParams", VariableKind.Uniform, `vec4<f32>`),
    // .x = screen width, .y = screen height.
    params2: new VariableMeta("uParams2", VariableKind.Uniform, `vec4<f32>`),
    invViewProj: new VariableMeta("uInvViewProj", VariableKind.Uniform, `mat4x4<f32>`),
    probeIn: new VariableMeta("probeIn", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "float",
    }),
    normalTex: new VariableMeta("normalTex", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "float",
    }),
    depthTex: new VariableMeta("depthTex", VariableKind.Texture, `texture_depth_2d`, {
      textureSampleType: "depth",
    }),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
const R: i32 = 2; // kernel radius (5×5)

const POSITION = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
  vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

struct VertexOutput { @builtin(position) position: vec4f };

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var out: VertexOutput;
  out.position = vec4f(POSITION[vertexIndex], 0.0, 1.0);
  return out;
}

fn unproject(ndc: vec3<f32>) -> vec3<f32> {
  let w = uInvViewProj * vec4<f32>(ndc, 1.0);
  return w.xyz / w.w;
}
fn world_at(px: vec2<i32>, screen: vec2<f32>) -> vec3<f32> {
  let d = textureLoad(depthTex, px, 0);
  let uv = (vec2<f32>(px) + vec2<f32>(0.5)) / screen;
  return unproject(vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, d));
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  let pc = vec2<i32>(floor(input.position.xy));
  let spacing = uParams.x;
  let normalSharp = uParams.y;
  let planeSigma = max(uParams.z, 1e-3);
  let blurSigma = max(uParams.w, 1e-3);
  let screen = uParams2.xy;

  let center = textureLoad(probeIn, pc, 0);
  if (center.a < 0.5) {
    return vec4f(0.0, 0.0, 0.0, 0.0);
  }
  let anchorC = min((vec2<f32>(pc) + vec2<f32>(0.5)) * spacing, screen - vec2<f32>(1.0));
  let cpx = vec2<i32>(anchorC);
  let Nc = normalize(textureLoad(normalTex, cpx, 0).rgb * 2.0 - 1.0);
  let posC = world_at(cpx, screen);

  let dims = vec2<i32>(textureDimensions(probeIn));
  let s2 = 2.0 * blurSigma * blurSigma;

  var acc = vec3<f32>(0.0);
  var wsum = 0.0;
  for (var j = -R; j <= R; j = j + 1) {
    for (var i = -R; i <= R; i = i + 1) {
      let q = pc + vec2<i32>(i, j);
      if (q.x < 0 || q.y < 0 || q.x >= dims.x || q.y >= dims.y) { continue; }
      let pd = textureLoad(probeIn, q, 0);
      if (pd.a < 0.5) { continue; }
      let aq = min((vec2<f32>(q) + vec2<f32>(0.5)) * spacing, screen - vec2<f32>(1.0));
      let qpx = vec2<i32>(aq);
      let nq = textureLoad(normalTex, qpx, 0);
      if (nq.a < 0.5) { continue; }
      let Nq = normalize(nq.rgb * 2.0 - 1.0);
      let posq = world_at(qpx, screen);

      let gk = exp(-f32(i * i + j * j) / s2);
      let nw = pow(max(0.0, dot(Nc, Nq)), normalSharp);
      let dw = exp(-abs(dot(Nc, posq - posC)) / planeSigma);
      let w = gk * nw * dw;
      acc = acc + pd.rgb * w;
      wsum = wsum + w;
    }
  }
  let outc = select(center.rgb, acc / wsum, wsum > 0.0);
  return vec4f(outc, 1.0);
}
`,
);
