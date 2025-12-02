import { query } from 'bitecs';
import { GPUShader } from '../../../../../../../renderer/src/WGSL/GPUShader.ts';
import { shaderMeta, MAX_EXPLOSION_COUNT } from './explosion.shader.ts';
import { getTypeTypedArray } from '../../../../../../../renderer/src/Shader';
import { projectionMatrix } from '../../../../../../../renderer/src/ECS/Systems/ResizeSystem.ts';
import { RenderDI } from '../../../../DI/RenderDI.ts';
import { GameDI } from '../../../../DI/GameDI.ts';
import { Explosion } from '../../../Components/Explosion.ts';
import { Progress } from '../../../Components/Progress.ts';
import { GlobalTransform } from '../../../../../../../renderer/src/ECS/Components/Transform.ts';

export function createDrawExplosionSystem({ device } = RenderDI, { world } = GameDI) {
    const gpuShader = new GPUShader(shaderMeta);
    const pipeline = gpuShader.getRenderPipeline(device, 'vs_main', 'fs_main', { withDepth: true });
    const bindGroup0 = gpuShader.getBindGroup(device, 0);
    const bindGroup1 = gpuShader.getBindGroup(device, 1);

    // Transform matrices for each explosion instance (mat4x4 = 16 floats)
    const transformBuffer = getTypeTypedArray(shaderMeta.uniforms.transform.type);
    // Explosion data: progress, seed for each explosion
    const explosionDataBuffer = getTypeTypedArray(shaderMeta.uniforms.explosionData.type);

    return (renderPass: GPURenderPassEncoder) => {
        const explosionEids = query(world, [Explosion, Progress, GlobalTransform]);
        const explosionCount = Math.min(explosionEids.length, MAX_EXPLOSION_COUNT);

        if (explosionCount === 0) {
            return;
        }

        // Collect data from GlobalTransform and Progress
        for (let i = 0; i < explosionCount; i++) {
            const eid = explosionEids[i];
            const globalMatrix = GlobalTransform.matrix.getBatch(eid);

            // Copy transform matrix directly from GlobalTransform
            const matOffset = i * 16;
            for (let j = 0; j < 16; j++) {
                transformBuffer[matOffset + j] = globalMatrix[j];
            }

            // Fill explosion data buffer: progress, seed
            const dataOffset = i * 2;
            explosionDataBuffer[dataOffset] = Progress.getProgress(eid);
            explosionDataBuffer[dataOffset + 1] = (eid * 0.1) % 1.0; // seed based on entity id
        }

        device.queue.writeBuffer(gpuShader.uniforms.projection.getGPUBuffer(device), 0, projectionMatrix as BufferSource);
        device.queue.writeBuffer(gpuShader.uniforms.transform.getGPUBuffer(device), 0, transformBuffer);
        device.queue.writeBuffer(gpuShader.uniforms.explosionData.getGPUBuffer(device), 0, explosionDataBuffer);

        renderPass.setBindGroup(0, bindGroup0);
        renderPass.setBindGroup(1, bindGroup1);
        renderPass.setPipeline(pipeline);
        renderPass.draw(6, explosionCount, 0, 0);
    };
}
