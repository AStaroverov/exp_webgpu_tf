import { ShaderMeta } from '../../../../../../../renderer/src/WGSL/ShaderMeta.ts';
import { wgsl } from '../../../../../../../renderer/src/WGSL/wgsl.ts';
import { VariableKind, VariableMeta } from '../../../../../../../renderer/src/Struct/VariableMeta.ts';

// Hand-tuned via the Lighting lil-gui panel; the directional source (sunAndSky) is the main light.
// Single floor for everything: objects are lit for real now (boundary dilation +
// translucent occluders), so a separate object ambient is no longer needed.
export const AMBIENT = 0.5;

export const shaderMeta = new ShaderMeta(
    {
        inputSampler: new VariableMeta('textureSampler', VariableKind.Sampler, `sampler`),
        sceneTexture: new VariableMeta('sceneTexture', VariableKind.Texture, `texture_2d<f32>`),
        radianceTexture: new VariableMeta('radianceTexture', VariableKind.Texture, `texture_2d<f32>`),
        emissionTexture: new VariableMeta('emissionTexture', VariableKind.Texture, `texture_2d<f32>`),
        ambient: new VariableMeta('uAmbient', VariableKind.Uniform, `f32`),
        objectLightRadius: new VariableMeta('uObjectLightRadius', VariableKind.Uniform, `f32`),
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

// 8-tap ring used to pull boundary light onto object pixels (occluders receive
// no RC radiance themselves — rays inside them immediately hit their own surface).
const DIR_COUNT = 8u;
const TAP_DIRS = array<vec2f, 8>(
    vec2f(1.0, 0.0), vec2f(0.7071, 0.7071),
    vec2f(0.0, 1.0), vec2f(-0.7071, 0.7071),
    vec2f(-1.0, 0.0), vec2f(-0.7071, -0.7071),
    vec2f(0.0, -1.0), vec2f(0.7071, -0.7071)
);

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  let scene = textureSample(sceneTexture, textureSampler, input.texCoord);
  var radiance = textureSample(radianceTexture, textureSampler, input.texCoord).rgb;
  // Coverage = object pixel (occluder/emitter); linear sampler softens the edge.
  let coverage = clamp(textureSample(emissionTexture, textureSampler, input.texCoord).a, 0.0, 1.0);

  // Object pixels: pull light from the boundary (dilated radiance). The max over
  // the ring keeps the light-facing side brighter than the shadow side.
  if (coverage > 0.0 && uObjectLightRadius > 0.0) {
    let texel = uObjectLightRadius / vec2f(textureDimensions(radianceTexture));
    var dilated = radiance;
    for (var i = 0u; i < DIR_COUNT; i = i + 1u) {
      let tap = textureSampleLevel(radianceTexture, textureSampler, input.texCoord + TAP_DIRS[i] * texel, 0.0).rgb;
      dilated = max(dilated, tap);
    }
    radiance = mix(radiance, dilated, coverage);
  }

  return vec4f(scene.rgb * (uAmbient + radiance), scene.a);
}
    `,
);
