import { ShaderMeta } from "../../../../../../../renderer/src/WGSL/ShaderMeta.ts";
import { wgsl } from "../../../../../../../renderer/src/WGSL/wgsl.ts";
import {
  VariableKind,
  VariableMeta,
} from "../../../../../../../renderer/src/Struct/VariableMeta.ts";

export const emissionTextureName = "emissionTexture";

export const shaderMeta = new ShaderMeta(
  {
    emissionTexture: new VariableMeta(emissionTextureName, VariableKind.Texture, `texture_2d<f32>`),
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
  let alpha = textureLoad(emissionTexture, vec2<i32>(input.position.xy), 0).a;
  return vec4f(input.texCoord * ceil(alpha), 0.0, 0.0);
}
    `,
);
