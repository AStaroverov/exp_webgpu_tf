import { GPUShader } from '../../../../../../../renderer/src/WGSL/GPUShader.ts';
import { shaderMeta } from './grass5.shader.ts';
import { getTypeTypedArray } from '../../../../../../../renderer/src/Shader';
import { sin } from '../../../../../../../../lib/math.ts';
import { random, randomSign } from '../../../../../../../../lib/random.ts';
import { RenderDI } from '../../../../DI/RenderDI.ts';
import { GameMap } from '../../../Entities/GameMap.ts';
import { GameDI } from '../../../../DI/GameDI.ts';

export function createDrawGrassSystem({ canvas, device } = RenderDI) {
    const gpuShader = new GPUShader(shaderMeta);
    const pipeline = gpuShader.getRenderPipeline(device, 'vs_main', 'fs_main', { withDepth: true });
    const bindGroup = gpuShader.getBindGroup(device, 0);

    const screenSizeBuffer = getTypeTypedArray(shaderMeta.uniforms.screenSize.type);
    const timeBuffer = getTypeTypedArray(shaderMeta.uniforms.time.type);
    const mapOffsetBuffer = getTypeTypedArray(shaderMeta.uniforms.mapOffset.type);
    const tileSizeBuffer = getTypeTypedArray(shaderMeta.uniforms.tileSize.type);
    const grassDensityBuffer = getTypeTypedArray(shaderMeta.uniforms.grassDensity.type);
    const windStrengthBuffer = getTypeTypedArray(shaderMeta.uniforms.windStrength.type);
    const windDirectionBuffer = getTypeTypedArray(shaderMeta.uniforms.windDirection.type);

    const colorCount = getTypeTypedArray(shaderMeta.uniforms.colorCount.type);
    const pixelSize = getTypeTypedArray(shaderMeta.uniforms.pixelSize.type);

    tileSizeBuffer[0] = 16;
    grassDensityBuffer[0] = 0.8;
    colorCount[0] = 4;
    pixelSize[0] = 1;

    device.queue.writeBuffer(gpuShader.uniforms.tileSize.getGPUBuffer(device), 0, tileSizeBuffer);
    device.queue.writeBuffer(gpuShader.uniforms.pixelSize.getGPUBuffer(device), 0, pixelSize);
    device.queue.writeBuffer(gpuShader.uniforms.colorCount.getGPUBuffer(device), 0, colorCount);
    device.queue.writeBuffer(gpuShader.uniforms.grassDensity.getGPUBuffer(device), 0, grassDensityBuffer);

    let time = 0;
    return (renderPass: GPURenderPassEncoder, delta: number) => {
        timeBuffer[0] = (time += delta / 100);
        screenSizeBuffer[0] = canvas.width;
        screenSizeBuffer[1] = canvas.height;
        
        // Convert world offset to pixel offset
        // canvas.width/GameDI.width = pixels per world unit
        const scaleX = canvas.width / GameDI.width;
        const scaleY = canvas.height / GameDI.height;
        mapOffsetBuffer[0] = GameMap.offsetX * scaleX;
        mapOffsetBuffer[1] = GameMap.offsetY * scaleY;
        
        windStrengthBuffer[0] = (1 + sin(windStrengthBuffer[0] + randomSign() * random() * delta / 100)) / 2;
        windDirectionBuffer[0] = sin(windDirectionBuffer[0] + randomSign() * random() * delta / 100);
        windDirectionBuffer[1] = sin(windDirectionBuffer[1] + randomSign() * random() * delta / 100);

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