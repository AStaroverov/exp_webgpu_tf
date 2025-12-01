import { query } from 'bitecs';
import { GPUShader } from '../../../../../../../renderer/src/WGSL/GPUShader.ts';
import { shaderMeta, MAX_MUZZLE_FLASH_COUNT } from './muzzleFlash.shader.ts';
import { getTypeTypedArray } from '../../../../../../../renderer/src/Shader';
import { RenderDI } from '../../../../DI/RenderDI.ts';
import { GameDI } from '../../../../DI/GameDI.ts';
import { MuzzleFlash } from '../../../Components/MuzzleFlash.ts';
import { ZIndex } from '../../../../consts.ts';

export function createDrawMuzzleFlashSystem({ canvas, device } = RenderDI, { world } = GameDI) {
    const gpuShader = new GPUShader(shaderMeta);
    const pipeline = gpuShader.getRenderPipeline(device, 'vs_main', 'fs_main', { withDepth: true });
    const bindGroup0 = gpuShader.getBindGroup(device, 0);
    const bindGroup1 = gpuShader.getBindGroup(device, 1);

    const screenSizeBuffer = getTypeTypedArray(shaderMeta.uniforms.screenSize.type);
    const flashCountBuffer = getTypeTypedArray(shaderMeta.uniforms.flashCount.type);
    const zIndexBuffer = getTypeTypedArray(shaderMeta.uniforms.zIndex.type);

    // Flash data: x, y, size, progress for each flash
    const flashDataBuffer = new Float32Array(MAX_MUZZLE_FLASH_COUNT * 4);
    const flashRotationBuffer = new Float32Array(MAX_MUZZLE_FLASH_COUNT);

    // Set constant zIndex
    zIndexBuffer[0] = ZIndex.MuzzleFlash;
    device.queue.writeBuffer(gpuShader.uniforms.zIndex.getGPUBuffer(device), 0, zIndexBuffer);

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

            // Fill buffer with flash data
            const offset = i * 4;
            flashDataBuffer[offset] = MuzzleFlash.x[eid];
            flashDataBuffer[offset + 1] = MuzzleFlash.y[eid];
            flashDataBuffer[offset + 2] = MuzzleFlash.size[eid];
            flashDataBuffer[offset + 3] = MuzzleFlash.getProgress(eid);
            flashRotationBuffer[i] = MuzzleFlash.rotation[eid];
        }

        screenSizeBuffer[0] = canvas.width;
        screenSizeBuffer[1] = canvas.height;
        flashCountBuffer[0] = flashCount;

        device.queue.writeBuffer(gpuShader.uniforms.screenSize.getGPUBuffer(device), 0, screenSizeBuffer);
        device.queue.writeBuffer(gpuShader.uniforms.flashCount.getGPUBuffer(device), 0, flashCountBuffer);
        device.queue.writeBuffer(gpuShader.uniforms.flashData.getGPUBuffer(device), 0, flashDataBuffer);
        device.queue.writeBuffer(gpuShader.uniforms.flashRotation.getGPUBuffer(device), 0, flashRotationBuffer);

        renderPass.setBindGroup(0, bindGroup0);
        renderPass.setBindGroup(1, bindGroup1);
        renderPass.setPipeline(pipeline);
        renderPass.draw(6, 1, 0, 0);
    };
}
