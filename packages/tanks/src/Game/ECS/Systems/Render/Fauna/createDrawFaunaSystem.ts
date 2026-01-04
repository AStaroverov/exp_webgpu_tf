import { GPUShader } from '../../../../../../../renderer/src/WGSL/GPUShader.ts';
import { shaderMeta } from './desert.shader.ts';
import { getTypeTypedArray } from '../../../../../../../renderer/src/Shader';
import { RenderDI } from '../../../../DI/RenderDI.ts';
import { cameraPosition } from 'renderer/src/ECS/Systems/ResizeSystem.ts';

export function createDrawFaunaSystem({ canvas, device } = RenderDI) {
    const gpuShader = new GPUShader(shaderMeta);
    const pipeline = gpuShader.getRenderPipeline(device, 'vs_main', 'fs_main', { withDepth: true });
    const bindGroup = gpuShader.getBindGroup(device, 0);

    const timeBuffer = getTypeTypedArray(shaderMeta.uniforms.time.type);
    const cameraPosBuffer = getTypeTypedArray(shaderMeta.uniforms.cameraPos.type);
    const screenSizeBuffer = getTypeTypedArray(shaderMeta.uniforms.screenSize.type);

    let time = 0;
    return (renderPass: GPURenderPassEncoder, delta: number) => {
        timeBuffer[0] = (time += delta / 1000); // Slower time for better control
        
        // Screen size in world units (1 CSS pixel = 1 world unit based on projection)
        screenSizeBuffer[0] = canvas.offsetWidth;
        screenSizeBuffer[1] = canvas.offsetHeight;
        
        // Camera position in world coordinates
        cameraPosBuffer[0] = cameraPosition.x;
        cameraPosBuffer[1] = cameraPosition.y;
        
        device.queue.writeBuffer(gpuShader.uniforms.time.getGPUBuffer(device), 0, timeBuffer);
        device.queue.writeBuffer(gpuShader.uniforms.cameraPos.getGPUBuffer(device), 0, cameraPosBuffer);
        device.queue.writeBuffer(gpuShader.uniforms.screenSize.getGPUBuffer(device), 0, screenSizeBuffer);

        renderPass.setBindGroup(0, bindGroup);
        renderPass.setPipeline(pipeline);
        renderPass.draw(6, 1, 0, 0);
    };
}



