import { GPUShader } from '../../../../../../../../renderer/src/WGSL/GPUShader.ts';
import { shaderMeta } from './sandstorm.shader.ts';
import { getTypeTypedArray } from '../../../../../../../../renderer/src/Shader';
import { RenderDI } from '../../../../../DI/RenderDI.ts';
import { GameMap } from '../../../../Entities/GameMap.ts';
import { GameDI } from '../../../../../DI/GameDI.ts';

export function createSandstormSystem({ canvas, device } = RenderDI) {
    const gpuShader = new GPUShader(shaderMeta);
    const pipeline = gpuShader.getRenderPipeline(device, 'vs_main', 'fs_main', { 
        withDepth: true, // Must match RenderPass depth attachment
    });
    const bindGroup = gpuShader.getBindGroup(device, 0);

    const screenSizeBuffer = getTypeTypedArray(shaderMeta.uniforms.screenSize.type);
    const timeBuffer = getTypeTypedArray(shaderMeta.uniforms.time.type);
    const windDirectionBuffer = getTypeTypedArray(shaderMeta.uniforms.windDirection.type);
    const mapOffsetBuffer = getTypeTypedArray(shaderMeta.uniforms.uMapOffset.type);

    let time = 0;
    return (renderPass: GPURenderPassEncoder, delta: number) => {
        time += delta / 1000;
        timeBuffer[0] = time;
        screenSizeBuffer[0] = canvas.width;
        screenSizeBuffer[1] = canvas.height;
        
        // Wind direction: blowing strongly to the right
        windDirectionBuffer[0] = 1.0;
        windDirectionBuffer[1] = 0.1;

        const scaleX = canvas.width / GameDI.width;
        const scaleY = canvas.height / GameDI.height;
        mapOffsetBuffer[0] = GameMap.offsetX * scaleX;
        mapOffsetBuffer[1] = GameMap.offsetY * scaleY;

        device.queue.writeBuffer(gpuShader.uniforms.time.getGPUBuffer(device), 0, timeBuffer);
        device.queue.writeBuffer(gpuShader.uniforms.screenSize.getGPUBuffer(device), 0, screenSizeBuffer);
        device.queue.writeBuffer(gpuShader.uniforms.windDirection.getGPUBuffer(device), 0, windDirectionBuffer);
        device.queue.writeBuffer(gpuShader.uniforms.uMapOffset.getGPUBuffer(device), 0, mapOffsetBuffer);

        // renderPass.setBindGroup(0, bindGroup);
        // renderPass.setPipeline(pipeline);
        // renderPass.draw(6, 1, 0, 0);
    };
}

