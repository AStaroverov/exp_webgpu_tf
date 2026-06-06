import { GPUShader } from './GPUShader.ts';
import { ShaderMeta } from './ShaderMeta.ts';
import { VariableKind, VariableMeta } from '../Struct/VariableMeta.ts';
import { wgsl } from './wgsl.ts';

const shaderMeta = new ShaderMeta(
    {
        inputSampler: new VariableMeta('inputSampler', VariableKind.Sampler, `sampler`),
        inputTexture: new VariableMeta('inputTexture', VariableKind.Texture, `texture_2d<f32>`),
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
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var output: VertexOutput;
  output.position = vec4f(POSITION[vertexIndex], 0.0, 1.0);
  output.texCoord = TEX_COORDS[vertexIndex];
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  return textureSample(inputTexture, inputSampler, input.texCoord);
}
    `,
);

/**
 * Final presentation: blits a texture to the swapchain. The ONLY place that touches
 * context.getCurrentTexture(), so the frame chain stays composable — call it last
 * with whatever texture is final this frame (also handy as a debug viewer for any
 * intermediate texture). Bind groups are cached per texture.
 */
export function createPresent(device: GPUDevice, context: GPUCanvasContext) {
    const sampler = device.createSampler({
        magFilter: 'nearest',
        minFilter: 'nearest',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
    });

    const cache = new Map<GPUTexture, { pipeline: GPURenderPipeline, bindGroup: GPUBindGroup }>();

    function getEntry(texture: GPUTexture) {
        let entry = cache.get(texture);
        if (entry === undefined) {
            shaderMeta.uniforms.inputSampler.setSampler(sampler);
            shaderMeta.uniforms.inputTexture.setTexture(texture);
            const shader = new GPUShader(shaderMeta);
            entry = {
                pipeline: shader.getRenderPipeline(device, 'vs_main', 'fs_main'),
                bindGroup: shader.getBindGroup(device, 0),
            };
            cache.set(texture, entry);
        }
        return entry;
    }

    return function present(encoder: GPUCommandEncoder, texture: GPUTexture) {
        const { pipeline, bindGroup } = getEntry(texture);

        const renderPass = encoder.beginRenderPass({
            colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                clearValue: [0.0, 0.0, 0.0, 0.0],
                loadOp: 'clear',
                storeOp: 'store',
            }],
        });
        renderPass.setPipeline(pipeline);
        renderPass.setBindGroup(0, bindGroup);
        renderPass.draw(6, 1, 0, 0);
        renderPass.end();
    };
}
