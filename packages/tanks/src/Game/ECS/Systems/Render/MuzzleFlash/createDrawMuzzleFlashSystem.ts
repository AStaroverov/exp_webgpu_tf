import { query } from 'bitecs';
import { GPUShader } from '../../../../../../../renderer/src/WGSL/GPUShader.ts';
import { shaderMeta, MAX_MUZZLE_FLASH_COUNT } from './muzzleFlash.shader.ts';
import { getTypeTypedArray } from '../../../../../../../renderer/src/Shader';
import { projectionMatrix } from '../../../../../../../renderer/src/ECS/Systems/ResizeSystem.ts';
import { RenderDI } from '../../../../DI/RenderDI.ts';
import { GameDI } from '../../../../DI/GameDI.ts';
import { MuzzleFlash } from '../../../Components/MuzzleFlash.ts';
import { ZIndex } from '../../../../consts.ts';

export function createDrawMuzzleFlashSystem({ device } = RenderDI, { world } = GameDI) {
    const gpuShader = new GPUShader(shaderMeta);
    const pipeline = gpuShader.getRenderPipeline(device, 'vs_main', 'fs_main', { withDepth: true });
    const bindGroup0 = gpuShader.getBindGroup(device, 0);
    const bindGroup1 = gpuShader.getBindGroup(device, 1);

    // Transform matrices for each flash instance (mat4x4 = 16 floats)
    const transformBuffer = getTypeTypedArray(shaderMeta.uniforms.transform.type);
    // Flash data: size, progress, seed, unused for each flash
    const flashDataBuffer = getTypeTypedArray(shaderMeta.uniforms.flashData.type);

    return (renderPass: GPURenderPassEncoder, delta: number) => {
        const flashEids = query(world, [MuzzleFlash]);
        const flashCount = Math.min(flashEids.length, MAX_MUZZLE_FLASH_COUNT);

        if (flashCount === 0) {
            return;
        }

        // Update flash age and collect data
        for (let i = 0; i < flashCount; i++) {
            const eid = flashEids[i];

            // Update age
            MuzzleFlash.updateAge(eid, delta);

            const x = MuzzleFlash.x[eid];
            const y = MuzzleFlash.y[eid];
            const rotation = MuzzleFlash.rotation[eid];
            const cosR = Math.cos(rotation);
            const sinR = Math.sin(rotation);

            // Build transform matrix (rotation + translation)
            // mat4x4 column-major:
            // [cosR, sinR, 0, 0]  col 0
            // [-sinR, cosR, 0, 0] col 1
            // [0, 0, 1, 0]        col 2
            // [x, y, z, 1]        col 3
            const matOffset = i * 16;
            transformBuffer[matOffset + 0] = cosR;
            transformBuffer[matOffset + 1] = sinR;
            transformBuffer[matOffset + 2] = 0;
            transformBuffer[matOffset + 3] = 0;
            transformBuffer[matOffset + 4] = -sinR;
            transformBuffer[matOffset + 5] = cosR;
            transformBuffer[matOffset + 6] = 0;
            transformBuffer[matOffset + 7] = 0;
            transformBuffer[matOffset + 8] = 0;
            transformBuffer[matOffset + 9] = 0;
            transformBuffer[matOffset + 10] = 1;
            transformBuffer[matOffset + 11] = 0;
            transformBuffer[matOffset + 12] = x;
            transformBuffer[matOffset + 13] = y;
            transformBuffer[matOffset + 14] = ZIndex.MuzzleFlash;
            transformBuffer[matOffset + 15] = 1;

            // Fill flash data buffer: size, progress, seed, unused
            const dataOffset = i * 4;
            flashDataBuffer[dataOffset] = MuzzleFlash.size[eid];
            flashDataBuffer[dataOffset + 1] = MuzzleFlash.getProgress(eid);
            flashDataBuffer[dataOffset + 2] = (eid * 0.1) % 1.0; // seed based on entity id
            flashDataBuffer[dataOffset + 3] = 0;
        }

        device.queue.writeBuffer(gpuShader.uniforms.projection.getGPUBuffer(device), 0, projectionMatrix as Float32Array);
        device.queue.writeBuffer(gpuShader.uniforms.transform.getGPUBuffer(device), 0, transformBuffer);
        device.queue.writeBuffer(gpuShader.uniforms.flashData.getGPUBuffer(device), 0, flashDataBuffer);

        renderPass.setBindGroup(0, bindGroup0);
        renderPass.setBindGroup(1, bindGroup1);
        renderPass.setPipeline(pipeline);
        renderPass.draw(6, flashCount, 0, 0);
    };
}
