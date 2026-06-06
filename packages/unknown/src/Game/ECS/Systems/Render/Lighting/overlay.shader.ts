import { ShaderMeta } from '../../../../../../../renderer/src/WGSL/ShaderMeta.ts';
import { wgsl } from '../../../../../../../renderer/src/WGSL/wgsl.ts';
import { VariableKind, VariableMeta } from '../../../../../../../renderer/src/Struct/VariableMeta.ts';

// Hand-tuned via the Lighting lil-gui panel; the directional source (sunAndSky) is the main light.
export const AMBIENT = 0.05;
// Fill light for object pixels: occluders receive no RC radiance (rays start inside
// them and immediately hit their own non-emissive surface), so without a separate
// floor they go pitch black when the ground ambient is lowered.
export const OBJECT_AMBIENT = 0.1;

export const shaderMeta = new ShaderMeta(
    {
        inputSampler: new VariableMeta('textureSampler', VariableKind.Sampler, `sampler`),
        sceneTexture: new VariableMeta('sceneTexture', VariableKind.Texture, `texture_2d<f32>`),
        radianceTexture: new VariableMeta('radianceTexture', VariableKind.Texture, `texture_2d<f32>`),
        emissionTexture: new VariableMeta('emissionTexture', VariableKind.Texture, `texture_2d<f32>`),
        ambient: new VariableMeta('uAmbient', VariableKind.Uniform, `f32`),
        objectAmbient: new VariableMeta('uObjectAmbient', VariableKind.Uniform, `f32`),
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
  let scene = textureSample(sceneTexture, textureSampler, input.texCoord);
  let radiance = textureSample(radianceTexture, textureSampler, input.texCoord).rgb;
  // Coverage = object pixel (occluder/emitter); linear sampler softens the edge.
  let coverage = clamp(textureSample(emissionTexture, textureSampler, input.texCoord).a, 0.0, 1.0);
  let ambient = mix(uAmbient, uObjectAmbient, coverage);
  return vec4f(scene.rgb * (ambient + radiance), scene.a);
}
    `,
);
