import { query } from 'bitecs';
import { GPUShader } from '../../../../../../../renderer/src/WGSL/GPUShader.ts';
import { shaderMeta, MAX_EXPLOSION_COUNT } from './explosion.shader.ts';
import { getTypeTypedArray } from '../../../../../../../renderer/src/Shader';
import { projectionMatrix } from '../../../../../../../renderer/src/ECS/Systems/ResizeSystem.ts';
import { RenderDI } from '../../../../DI/RenderDI.ts';
import { GameDI } from '../../../../DI/GameDI.ts';
import { Explosion } from '../../../Components/Explosion.ts';
import { ZIndex } from '../../../../consts.ts';

export function createDrawExplosionSystem({ device } = RenderDI, { world } = GameDI) {
    const gpuShader = new GPUShader(shaderMeta);
    const pipeline = gpuShader.getRenderPipeline(device, 'vs_main', 'fs_main', { withDepth: true });
    const bindGroup0 = gpuShader.getBindGroup(device, 0);
    const bindGroup1 = gpuShader.getBindGroup(device, 1);

    // Transform matrices for each explosion instance (mat4x4 = 16 floats)
    const transformBuffer = getTypeTypedArray(shaderMeta.uniforms.transform.type);
    // Explosion data: size, progress, seed, unused for each explosion
    const explosionDataBuffer = getTypeTypedArray(shaderMeta.uniforms.explosionData.type);

    return (renderPass: GPURenderPassEncoder, delta: number) => {
        const explosionEids = query(world, [Explosion]);
        const explosionCount = Math.min(explosionEids.length, MAX_EXPLOSION_COUNT);

        if (explosionCount === 0) {
            return;
        }

        // Update explosion age and collect data
        for (let i = 0; i < explosionCount; i++) {
            const eid = explosionEids[i];

            // Update age
            Explosion.updateAge(eid, delta);

            const x = Explosion.x[eid];
            const y = Explosion.y[eid];

            // Build transform matrix (no rotation for explosion - radial effect)
            // mat4x4 column-major:
            // [1, 0, 0, 0]  col 0
            // [0, 1, 0, 0]  col 1
            // [0, 0, 1, 0]  col 2
            // [x, y, z, 1]  col 3
            const matOffset = i * 16;
            transformBuffer[matOffset + 0] = 1;
            transformBuffer[matOffset + 1] = 0;
            transformBuffer[matOffset + 2] = 0;
            transformBuffer[matOffset + 3] = 0;
            transformBuffer[matOffset + 4] = 0;
            transformBuffer[matOffset + 5] = 1;
            transformBuffer[matOffset + 6] = 0;
            transformBuffer[matOffset + 7] = 0;
            transformBuffer[matOffset + 8] = 0;
            transformBuffer[matOffset + 9] = 0;
            transformBuffer[matOffset + 10] = 1;
            transformBuffer[matOffset + 11] = 0;
            transformBuffer[matOffset + 12] = x;
            transformBuffer[matOffset + 13] = y;
            transformBuffer[matOffset + 14] = ZIndex.Explosion;
            transformBuffer[matOffset + 15] = 1;

            // Fill explosion data buffer: size, progress, seed, unused
            const dataOffset = i * 4;
            explosionDataBuffer[dataOffset] = Explosion.size[eid];
            explosionDataBuffer[dataOffset + 1] = Explosion.getProgress(eid);
            explosionDataBuffer[dataOffset + 2] = (eid * 0.1) % 1.0; // seed based on entity id
            explosionDataBuffer[dataOffset + 3] = 0;
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
