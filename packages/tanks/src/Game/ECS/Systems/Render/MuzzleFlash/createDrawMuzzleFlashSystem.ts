import { query } from 'bitecs';
import { GPUShader } from '../../../../../../../renderer/src/WGSL/GPUShader.ts';
import { shaderMeta, MAX_MUZZLE_FLASH_COUNT } from './muzzleFlash.shader.ts';
import { getTypeTypedArray } from '../../../../../../../renderer/src/Shader';
import { projectionMatrix } from '../../../../../../../renderer/src/ECS/Systems/ResizeSystem.ts';
import { RenderDI } from '../../../../DI/RenderDI.ts';
import { GameDI } from '../../../../DI/GameDI.ts';
import { MuzzleFlash } from '../../../Components/MuzzleFlash.ts';
import { Progress } from '../../../Components/Progress.ts';
import { GlobalTransform } from '../../../../../../../renderer/src/ECS/Components/Transform.ts';

export function createDrawMuzzleFlashSystem({ device } = RenderDI, { world } = GameDI) {
    const gpuShader = new GPUShader(shaderMeta);
    const pipeline = gpuShader.getRenderPipeline(device, 'vs_main', 'fs_main', { withDepth: true });
    const bindGroup0 = gpuShader.getBindGroup(device, 0);
    const bindGroup1 = gpuShader.getBindGroup(device, 1);

    // Transform matrices for each flash instance (mat4x4 = 16 floats)
    const transformBuffer = getTypeTypedArray(shaderMeta.uniforms.transform.type);
    // Flash data: progress, seed for each flash
    const flashDataBuffer = getTypeTypedArray(shaderMeta.uniforms.flashData.type);

    return (renderPass: GPURenderPassEncoder) => {
        const flashEids = query(world, [MuzzleFlash, Progress, GlobalTransform]);
        const flashCount = Math.min(flashEids.length, MAX_MUZZLE_FLASH_COUNT);

        if (flashCount === 0) {
            return;
        }

        // Collect data from GlobalTransform and Progress
        for (let i = 0; i < flashCount; i++) {
            const eid = flashEids[i];
            const globalMatrix = GlobalTransform.matrix.getBatch(eid);

            // Copy transform matrix directly from GlobalTransform
            const matOffset = i * 16;
            for (let j = 0; j < 16; j++) {
                transformBuffer[matOffset + j] = globalMatrix[j];
            }

            // Fill flash data buffer: progress, seed
            const dataOffset = i * 2;
            flashDataBuffer[dataOffset] = Progress.getProgress(eid);
            flashDataBuffer[dataOffset + 1] = (eid * 0.1) % 1.0; // seed based on entity id
        }

        device.queue.writeBuffer(gpuShader.uniforms.projection.getGPUBuffer(device), 0, projectionMatrix as BufferSource);
        device.queue.writeBuffer(gpuShader.uniforms.transform.getGPUBuffer(device), 0, transformBuffer);
        device.queue.writeBuffer(gpuShader.uniforms.flashData.getGPUBuffer(device), 0, flashDataBuffer);

        renderPass.setBindGroup(0, bindGroup0);
        renderPass.setBindGroup(1, bindGroup1);
        renderPass.setPipeline(pipeline);
        renderPass.draw(6, flashCount, 0, 0);
    };
}
