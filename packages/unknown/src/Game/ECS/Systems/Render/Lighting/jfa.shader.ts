import { ShaderMeta } from "../../../../../../../renderer/src/WGSL/ShaderMeta.ts";
import { wgsl } from "../../../../../../../renderer/src/WGSL/wgsl.ts";
import {
  VariableKind,
  VariableMeta,
} from "../../../../../../../renderer/src/Struct/VariableMeta.ts";

export const inputTextureName = "inputTexture";
export const oneOverSizeName = "oneOverSize";
export const uOffsetName = "uOffset";

export const shaderMeta = new ShaderMeta(
  {
    inputTexture: new VariableMeta(inputTextureName, VariableKind.Texture, `texture_2d<f32>`),
    oneOverSize: new VariableMeta(oneOverSizeName, VariableKind.Uniform, `vec2<f32>`),
    uOffset: new VariableMeta(uOffsetName, VariableKind.Uniform, `f32`),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
const POSITION = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f(1.0, -1.0),
    vec2f(1.0, 1.0),
    vec2f(-1.0, -1.0),
    vec2f(1.0, 1.0),
    vec2f(-1.0, 1.0)
  );

const TEX_COORDS = array<vec2f, 6>(
    vec2f(0.0, 1.0),
    vec2f(1.0, 1.0),
    vec2f(1.0, 0.0),
    vec2f(0.0, 1.0),
    vec2f(1.0, 0.0),
    vec2f(0.0, 0.0)
  );

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) texCoord: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {var output: VertexOutput;
  output.position = vec4f(POSITION[vertexIndex], 0.0, 1.0);
  output.texCoord = TEX_COORDS[vertexIndex];
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  let vUv = input.texCoord;
  let resolution = 1.0 / oneOverSize;

  var nearestSeed = vec2f(-1.0);
  var nearestDist = 999999.9;
  let pre = uOffset * oneOverSize;

  let centerValue = textureLoad(inputTexture, vec2<i32>(vUv * resolution), 0).xy;
  if (centerValue.x > 0.0 || centerValue.y > 0.0) {
    let diff = centerValue - vUv;
    let dist = dot(diff, diff);
    if (dist < nearestDist) {
      nearestDist = dist;
      nearestSeed = centerValue;
    }
  }

  for (var y = -1.0; y <= 1.0; y += 1.0) {
    for (var x = -1.0; x <= 1.0; x += 1.0) {
      if (x == 0.0 && y == 0.0) { continue; }
      let sampleUV = vUv + vec2f(x, y) * pre;

      if (sampleUV.x < 0.0 || sampleUV.x > 1.0 || sampleUV.y < 0.0 || sampleUV.y > 1.0) { continue; }

      let sampleValue = textureLoad(inputTexture, vec2<i32>(sampleUV * resolution), 0).xy;
      if (sampleValue.x > 0.0 || sampleValue.y > 0.0) {
        let diff = sampleValue - vUv;
        let dist = dot(diff, diff);
        if (dist < nearestDist) {
          nearestDist = dist;
          nearestSeed = sampleValue;
        }
      }
    }
  }

  return vec4f(nearestSeed, 0.0, 0.0);
}
    `,
);
