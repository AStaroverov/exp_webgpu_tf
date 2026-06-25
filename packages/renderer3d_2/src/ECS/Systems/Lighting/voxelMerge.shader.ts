import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";

// Stage 2.3b/c — cascade merge with geometric weighting. mergedAtlas[c] = rawAtlas[c] +
// visibility[c] * farContribution, where farContribution is the cascade c+1 radiance in the
// matching direction, interpolated SPATIALLY (over the 4 nearest c+1 probes), ANGULARLY
// (average of the 4 child directions), and weighted by NORMAL SIMILARITY (a c+1 probe on a
// different surface is rejected → no cross-cascade leak). Run top-down (c = N-2 … 0).

export const shaderMeta = new ShaderMeta(
  {
    // [0] = (spacingC, dirsSideC, spacingUp, dirsSideUp),
    // [1] = (screenW, screenH, probesXUp, probesYUp).
    merge: new VariableMeta("uMerge", VariableKind.Uniform, `array<vec4<f32>, 2>`),
    // .x = normalSharpness (shared across merges).
    params: new VariableMeta("uParams", VariableKind.Uniform, `vec4<f32>`),
    rawTex: new VariableMeta("rawTex", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "float",
    }),
    upperTex: new VariableMeta("upperTex", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "float",
    }),
    normalTex: new VariableMeta("normalTex", VariableKind.Texture, `texture_2d<f32>`, {
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

fn normal_at(anchor: vec2<f32>, screen: vec2<f32>) -> vec4<f32> {
  let apx = vec2<i32>(min(anchor, screen - vec2<f32>(1.0)));
  return textureLoad(normalTex, apx, 0);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  let spacingC = uMerge[0].x;
  let dirsSideC = i32(uMerge[0].y);
  let spacingUp = uMerge[0].z;
  let dirsSideUp = i32(uMerge[0].w);
  let screen = uMerge[1].xy;
  let probesUp = vec2<i32>(i32(uMerge[1].z), i32(uMerge[1].w));
  let normalSharp = uParams.x;

  let texel = vec2<i32>(floor(input.position.xy));
  let probeCoordC = texel / dirsSideC;
  let dirCoordC = texel - probeCoordC * dirsSideC;

  let raw = textureLoad(rawTex, texel, 0);

  // This probe's surface normal + its position in the (coarser) c+1 grid.
  let anchorC = (vec2<f32>(probeCoordC) + vec2<f32>(0.5)) * spacingC;
  let nC = normal_at(anchorC, screen);
  let Nc = normalize(nC.rgb * 2.0 - 1.0);

  let fp = anchorC / spacingUp - vec2<f32>(0.5);
  let baseP = vec2<i32>(floor(fp));
  let fr = fp - floor(fp);
  let childBase = dirCoordC * 2;

  var far = vec3<f32>(0.0);
  var wsum = 0.0;
  for (var j = 0; j < 2; j = j + 1) {
    for (var i = 0; i < 2; i = i + 1) {
      let np = clamp(baseP + vec2<i32>(i, j), vec2<i32>(0), probesUp - vec2<i32>(1));
      let bw = select(1.0 - fr.x, fr.x, i == 1) * select(1.0 - fr.y, fr.y, j == 1);
      // Geometric weight: reject c+1 probes on a different surface.
      let anchorUp = (vec2<f32>(np) + vec2<f32>(0.5)) * spacingUp;
      let nUp = normal_at(anchorUp, screen);
      let gw = select(0.0, pow(max(0.0, dot(Nc, normalize(nUp.rgb * 2.0 - 1.0))), normalSharp), nUp.a > 0.5);
      let w = bw * gw;
      // Angular average of the 4 child directions at this spatial neighbour.
      var ang = vec3<f32>(0.0);
      for (var cj = 0; cj < 2; cj = cj + 1) {
        for (var ci = 0; ci < 2; ci = ci + 1) {
          let cd = childBase + vec2<i32>(ci, cj);
          ang = ang + textureLoad(upperTex, np * dirsSideUp + cd, 0).rgb;
        }
      }
      far = far + ang * 0.25 * w;
      wsum = wsum + w;
    }
  }
  far = select(vec3<f32>(0.0), far / wsum, wsum > 0.0);

  let merged = raw.rgb + raw.a * far;
  return vec4f(merged, raw.a);
}
`,
);
