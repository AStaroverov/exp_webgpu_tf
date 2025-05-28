import { GPUShader } from '../../../../../../../../renderer/src/WGSL/GPUShader.ts';
import { shaderMeta } from './shader.ts';

export function createPostEffect(device: GPUDevice, context: GPUCanvasContext, sourceTexture: GPUTexture) {
    const sampler = device.createSampler({
        magFilter: 'nearest',
        minFilter: 'nearest',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
    });

    shaderMeta.uniforms.inputSampler.setSampler(sampler);
    shaderMeta.uniforms.inputTexture.setTexture(sourceTexture);

    const gpuShader = new GPUShader(shaderMeta);
    const pipeline = gpuShader.getRenderPipeline(device, 'vs_main', 'fs_main');
    const bindGroup = gpuShader.getBindGroup(device, 0);

    return (commandEncoder: GPUCommandEncoder) => {
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [
                {
                    view: context.getCurrentTexture().createView(),
                    clearValue: [0.0, 0.0, 0.0, 0.0],
                    loadOp: 'clear',
                    storeOp: 'store',
                },
            ],
        });

        renderPass.setPipeline(pipeline);
        renderPass.setBindGroup(0, bindGroup);
        renderPass.draw(6, 1, 0, 0);
        renderPass.end();
    };
}

