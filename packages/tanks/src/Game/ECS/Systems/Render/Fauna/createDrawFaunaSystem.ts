import { GPUShader } from '../../../../../../../renderer/src/WGSL/GPUShader.ts';
import { shaderMeta } from './desert.shader.ts';
import { getTypeTypedArray } from '../../../../../../../renderer/src/Shader';
import { sin } from '../../../../../../../../lib/math.ts';
import { RenderDI } from '../../../../DI/RenderDI.ts';
import { GameMap } from '../../../Entities/GameMap.ts';
import { GameDI } from '../../../../DI/GameDI.ts';

export function createDrawFaunaSystem({ canvas, device } = RenderDI) {
    const gpuShader = new GPUShader(shaderMeta);
    const pipeline = gpuShader.getRenderPipeline(device, 'vs_main', 'fs_main', { withDepth: true });
    const bindGroup = gpuShader.getBindGroup(device, 0);

    const screenSizeBuffer = getTypeTypedArray(shaderMeta.uniforms.screenSize.type);
    const timeBuffer = getTypeTypedArray(shaderMeta.uniforms.time.type);
    const mapOffsetBuffer = getTypeTypedArray(shaderMeta.uniforms.mapOffset.type);
    const tileSizeBuffer = getTypeTypedArray(shaderMeta.uniforms.tileSize.type);
    const densityBuffer = getTypeTypedArray(shaderMeta.uniforms.density.type);
    const windStrengthBuffer = getTypeTypedArray(shaderMeta.uniforms.windStrength.type);
    const windDirectionBuffer = getTypeTypedArray(shaderMeta.uniforms.windDirection.type);
    const pixelSize = getTypeTypedArray(shaderMeta.uniforms.pixelSize.type);

    tileSizeBuffer[0] = 32; // Larger tiles for desert fauna
    densityBuffer[0] = 0.4; // Slightly lower density
    pixelSize[0] = 1;

    device.queue.writeBuffer(gpuShader.uniforms.tileSize.getGPUBuffer(device), 0, tileSizeBuffer);
    device.queue.writeBuffer(gpuShader.uniforms.pixelSize.getGPUBuffer(device), 0, pixelSize);
    device.queue.writeBuffer(gpuShader.uniforms.density.getGPUBuffer(device), 0, densityBuffer);

    let time = 0;
    return (renderPass: GPURenderPassEncoder, delta: number) => {
        timeBuffer[0] = (time += delta / 1000); // Slower time for better control
        screenSizeBuffer[0] = canvas.width;
        screenSizeBuffer[1] = canvas.height;
        
        const scaleX = canvas.width / GameDI.width;
        const scaleY = canvas.height / GameDI.height;
        mapOffsetBuffer[0] = GameMap.offsetX * scaleX;
        mapOffsetBuffer[1] = GameMap.offsetY * scaleY;
        
        // Dynamic wind for tumbleweeds
        windStrengthBuffer[0] = (1 + sin(time * 0.5)) / 2;
        windDirectionBuffer[0] = 1.0; // Predominantly blowing right
        windDirectionBuffer[1] = 0.2 * sin(time * 0.3);

        device.queue.writeBuffer(gpuShader.uniforms.time.getGPUBuffer(device), 0, timeBuffer);
        device.queue.writeBuffer(gpuShader.uniforms.screenSize.getGPUBuffer(device), 0, screenSizeBuffer);
        device.queue.writeBuffer(gpuShader.uniforms.mapOffset.getGPUBuffer(device), 0, mapOffsetBuffer);
        device.queue.writeBuffer(gpuShader.uniforms.windStrength.getGPUBuffer(device), 0, windStrengthBuffer);
        device.queue.writeBuffer(gpuShader.uniforms.windDirection.getGPUBuffer(device), 0, windDirectionBuffer);

        renderPass.setBindGroup(0, bindGroup);
        renderPass.setPipeline(pipeline);
        renderPass.draw(6, 1, 0, 0);
    };
}



