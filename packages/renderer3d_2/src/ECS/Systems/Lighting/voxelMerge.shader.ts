import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";

// Stage 2.3b — cascade merge. Writes mergedAtlas[c] = rawAtlas[c] + visibility[c] *
// farContribution, where farContribution is the cascade c+1 radiance in the matching
// direction, interpolated SPATIALLY (bilinear over the 4 nearest c+1 probes) and ANGULARLY
// (average of the 4 child directions). Run top-down (c = N-2 … 0). The "upper" texture is
// mergedAtlas[c+1] (or rawAtlas[N-1] for the topmost merge).
//
// Atlas layout: probe (i,j) of a cascade with dirsPerSide D occupies a D×D block; atlas
// size is constant across cascades. A c-direction (du,dv) refines into the c+1 children
// (2du..2du+1, 2dv..2dv+1) — consistent octahedral parametrisation makes them cluster.

export const shaderMeta = new ShaderMeta(
  {
    // [0] = (spacingC, dirsSideC, spacingUp, dirsSideUp),
    // [1] = (screenW, screenH, probesXUp, probesYUp).
    merge: new VariableMeta("uMerge", VariableKind.Uniform, `array<vec4<f32>, 2>`),
    rawTex: new VariableMeta("rawTex", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "float",
    }),
    upperTex: new VariableMeta("upperTex", VariableKind.Texture, `texture_2d<f32>`, {
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

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  let spacingC = uMerge[0].x;
  let dirsSideC = i32(uMerge[0].y);
  let spacingUp = uMerge[0].z;
  let dirsSideUp = i32(uMerge[0].w);
  let probesUp = vec2<i32>(i32(uMerge[1].z), i32(uMerge[1].w));

  let texel = vec2<i32>(floor(input.position.xy));
  let probeCoordC = texel / dirsSideC;
  let dirCoordC = texel - probeCoordC * dirsSideC;

  let raw = textureLoad(rawTex, texel, 0);

  // This probe's screen anchor → fractional probe coords in the (coarser) c+1 grid.
  let anchor = (vec2<f32>(probeCoordC) + vec2<f32>(0.5)) * spacingC;
  let fp = anchor / spacingUp - vec2<f32>(0.5);
  let baseP = vec2<i32>(floor(fp));
  let fr = fp - floor(fp);

  // The 4 c+1 child directions of this c direction.
  let childBase = dirCoordC * 2;

  var far = vec3<f32>(0.0);
  var wsum = 0.0;
  for (var j = 0; j < 2; j = j + 1) {
    for (var i = 0; i < 2; i = i + 1) {
      let np = clamp(baseP + vec2<i32>(i, j), vec2<i32>(0), probesUp - vec2<i32>(1));
      let bw = select(1.0 - fr.x, fr.x, i == 1) * select(1.0 - fr.y, fr.y, j == 1);
      // Angular average of the 4 child directions at this spatial neighbour.
      var ang = vec3<f32>(0.0);
      for (var cj = 0; cj < 2; cj = cj + 1) {
        for (var ci = 0; ci < 2; ci = ci + 1) {
          let cd = childBase + vec2<i32>(ci, cj);
          ang = ang + textureLoad(upperTex, np * dirsSideUp + cd, 0).rgb;
        }
      }
      far = far + ang * 0.25 * bw;
      wsum = wsum + bw;
    }
  }
  far = far / max(wsum, 1e-4);

  let merged = raw.rgb + raw.a * far;
  return vec4f(merged, raw.a);
}
`,
);
