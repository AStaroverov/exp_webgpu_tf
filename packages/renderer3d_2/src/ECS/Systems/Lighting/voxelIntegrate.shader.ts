import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";

// Stage 2.3a — integrate. Reads the (merged) probe atlas and collapses each probe's
// DIR_W×DIR_W directional tile to a single irradiance value, cosine-weighted by the
// probe's surface normal (from the G-buffer). Output = probeIrr (one texel per probe),
// which the resolve pass bilinearly samples. Separating this keeps the resolve cheap.

export const shaderMeta = new ShaderMeta(
  {
    // .x = probe spacing (px), .y = cascade-0 dirs/side (DIR_W).
    params: new VariableMeta("uParams", VariableKind.Uniform, `vec4<f32>`),
    // .x = screen width, .y = screen height.
    params2: new VariableMeta("uParams2", VariableKind.Uniform, `vec4<f32>`),
    normalTex: new VariableMeta("normalTex", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "float",
    }),
    cascadeAtlas: new VariableMeta("cascadeAtlas", VariableKind.Texture, `texture_2d<f32>`, {
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

struct VertexOutput { @builtin(position) position: vec4f };

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var out: VertexOutput;
  out.position = vec4f(POSITION[vertexIndex], 0.0, 1.0);
  return out;
}

fn oct_decode(e: vec2<f32>) -> vec3<f32> {
  var v = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
  if (v.z < 0.0) {
    let s = vec2<f32>(select(-1.0, 1.0, v.x >= 0.0), select(-1.0, 1.0, v.y >= 0.0));
    v = vec3<f32>((1.0 - abs(v.yx)) * s, v.z);
  }
  return normalize(v);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  let probeCoord = vec2<i32>(floor(input.position.xy));
  let spacing = uParams.x;
  let DIR_W = i32(uParams.y);
  let screen = uParams2.xy;
  let anchorPx = min((vec2<f32>(probeCoord) + vec2<f32>(0.5)) * spacing, screen - vec2<f32>(1.0));

  let n = textureLoad(normalTex, vec2<i32>(anchorPx), 0);
  if (n.a < 0.5) {
    return vec4f(0.0, 0.0, 0.0, 0.0);
  }
  let N = normalize(n.rgb * 2.0 - 1.0);

  let base = probeCoord * DIR_W;
  var acc = vec3<f32>(0.0);
  var wsum = 0.0;
  for (var v = 0; v < DIR_W; v = v + 1) {
    for (var u = 0; u < DIR_W; u = u + 1) {
      let e = (vec2<f32>(f32(u), f32(v)) + vec2<f32>(0.5)) / f32(DIR_W) * 2.0 - 1.0;
      let dir = oct_decode(e);
      let w = max(0.0, dot(dir, N));
      if (w > 0.0) {
        let rad = textureLoad(cascadeAtlas, base + vec2<i32>(u, v), 0).rgb;
        acc = acc + rad * w;
        wsum = wsum + w;
      }
    }
  }
  let irr = acc / max(wsum, 1e-4);
  return vec4f(irr, 1.0);
}
`,
);
