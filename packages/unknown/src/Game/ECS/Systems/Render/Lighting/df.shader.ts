import { ShaderMeta } from '../../../../../../../renderer/src/WGSL/ShaderMeta.ts';
import { wgsl } from '../../../../../../../renderer/src/WGSL/wgsl.ts';
import { VariableKind, VariableMeta } from '../../../../../../../renderer/src/Struct/VariableMeta.ts';

export const jfaTextureName = 'jfaTexture';

export const shaderMeta = new ShaderMeta(
    {
        jfaTexture: new VariableMeta(jfaTextureName, VariableKind.Texture, `texture_2d<f32>`),
    },
    {},
    // language=WGSL
    wgsl/* wgsl */ `
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
  let resolution = vec2f(textureDimensions(jfaTexture));
  let texel = vec2<i32>(vUv * resolution);
  let nearestSeed = textureLoad(jfaTexture, texel, 0).xy;
  let dist = clamp(distance(vUv, nearestSeed), 0.0, 1.0);
  return vec4f(dist, 0.0, 0.0, 0.0);
}
    `,
);
