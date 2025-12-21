import { query } from 'bitecs';
import { GPUShader } from 'renderer/src/WGSL/GPUShader.ts';
import { shaderMeta, MAX_VFX_COUNT } from './vfx.shader.ts';
import { getTypeTypedArray } from 'renderer/src/Shader/index.ts';
import { projectionMatrix } from 'renderer/src/ECS/Systems/ResizeSystem.ts';
import { RenderDI } from '../../../../DI/RenderDI.ts';
import { GameDI } from '../../../../DI/GameDI.ts';
import { VFX, VFXType, VFXTypeValue } from '../../../Components/VFX.ts';
import { Progress } from '../../../Components/Progress.ts';
import { GlobalTransform } from 'renderer/src/ECS/Components/Transform.ts';

// Max radius calculation per effect type
const getMaxRadius: Record<VFXTypeValue, (progress: number) => number> = {
    [VFXType.ExhaustSmoke]: (progress) => {
        const expandFactor = 1.0 + progress * 4.0;
        return 1.0 * expandFactor;
    },
    [VFXType.Explosion]: (progress) => {
        const expandFactor = 1.0 + progress * 0.5;
        return expandFactor * 1.5;
    },
    [VFXType.HitFlash]: (progress) => {
        const expandedSize = 1.0 * (1.0 + progress * 0.5);
        return expandedSize * 2.5;
    },
    [VFXType.MuzzleFlash]: (progress) => {
        const expandedSize = 1.0 * (0.6 + progress * 1.44);
        return expandedSize * 2.0;
    },
};

// Seed multiplier per effect type
const seedMultiplier: Record<VFXTypeValue, number> = {
    [VFXType.ExhaustSmoke]: 0.137,
    [VFXType.Explosion]: 0.1,
    [VFXType.HitFlash]: 0.1,
    [VFXType.MuzzleFlash]: 0.1,
};

export function createDrawVFXSystem({ device } = RenderDI, { world } = GameDI) {
    const gpuShader = new GPUShader(shaderMeta);
    const pipeline = gpuShader.getRenderPipeline(device, 'vs_main', 'fs_main', { withDepth: true });
    const bindGroup0 = gpuShader.getBindGroup(device, 0);
    const bindGroup1 = gpuShader.getBindGroup(device, 1);

    // Transform matrices for each VFX instance (mat4x4 = 16 floats)
    const transformBuffer = getTypeTypedArray(shaderMeta.uniforms.transform.type);
    // VFX data: progress, seed, effectType, maxRadius
    const vfxDataBuffer = getTypeTypedArray(shaderMeta.uniforms.vfxData.type);

    return (renderPass: GPURenderPassEncoder) => {
        const eids = query(world, [VFX, Progress, GlobalTransform]);
        const count = Math.min(eids.length, MAX_VFX_COUNT);

        if (count === 0) {
            return;
        }

        for (let i = 0; i < count; i++) {
            const eid = eids[i];
            const effectType = VFX.type[eid] as VFXTypeValue;
            const progress = Progress.getProgress(eid);
            const globalMatrix = GlobalTransform.matrix.getBatch(eid);

            // Copy transform matrix
            transformBuffer.set(globalMatrix, i * globalMatrix.length);

            // Fill VFX data: progress, seed, effectType, maxRadius
            const dataOffset = i * 4;
            vfxDataBuffer[dataOffset] = progress;
            vfxDataBuffer[dataOffset + 1] = (eid * seedMultiplier[effectType]) % 1.0;
            vfxDataBuffer[dataOffset + 2] = effectType;
            vfxDataBuffer[dataOffset + 3] = getMaxRadius[effectType](progress);
        }

        device.queue.writeBuffer(gpuShader.uniforms.projection.getGPUBuffer(device), 0, projectionMatrix as BufferSource);
        device.queue.writeBuffer(gpuShader.uniforms.transform.getGPUBuffer(device), 0, transformBuffer);
        device.queue.writeBuffer(gpuShader.uniforms.vfxData.getGPUBuffer(device), 0, vfxDataBuffer);

        renderPass.setBindGroup(0, bindGroup0);
        renderPass.setBindGroup(1, bindGroup1);
        renderPass.setPipeline(pipeline);
        renderPass.draw(6, count, 0, 0);
    };
}
