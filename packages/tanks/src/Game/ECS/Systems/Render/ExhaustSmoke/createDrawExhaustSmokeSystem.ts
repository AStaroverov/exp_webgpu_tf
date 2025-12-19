import { query } from 'bitecs';
import { GPUShader } from '../../../../../../../renderer/src/WGSL/GPUShader.ts';
import { shaderMeta, MAX_EXHAUST_SMOKE_COUNT } from './exhaustSmoke.shader.ts';
import { getTypeTypedArray } from '../../../../../../../renderer/src/Shader';
import { projectionMatrix } from '../../../../../../../renderer/src/ECS/Systems/ResizeSystem.ts';
import { RenderDI } from '../../../../DI/RenderDI.ts';
import { GameDI } from '../../../../DI/GameDI.ts';
import { ExhaustSmoke } from '../../../Components/ExhaustSmoke.ts';
import { Progress } from '../../../Components/Progress.ts';
import { GlobalTransform } from '../../../../../../../renderer/src/ECS/Components/Transform.ts';

export function createDrawExhaustSmokeSystem({ device } = RenderDI, { world } = GameDI) {
    const gpuShader = new GPUShader(shaderMeta);
    const pipeline = gpuShader.getRenderPipeline(device, 'vs_main', 'fs_main', { withDepth: true });
    const bindGroup0 = gpuShader.getBindGroup(device, 0);
    const bindGroup1 = gpuShader.getBindGroup(device, 1);

    // Transform matrices for each smoke instance (mat4x4 = 16 floats)
    const transformBuffer = getTypeTypedArray(shaderMeta.uniforms.transform.type);
    // Smoke data: progress, seed for each smoke particle
    const smokeDataBuffer = getTypeTypedArray(shaderMeta.uniforms.smokeData.type);

    return (renderPass: GPURenderPassEncoder) => {
        const smokeEids = query(world, [ExhaustSmoke, Progress, GlobalTransform]);
        const smokeCount = Math.min(smokeEids.length, MAX_EXHAUST_SMOKE_COUNT);

        if (smokeCount === 0) {
            return;
        }

        // Collect data from GlobalTransform and Progress
        for (let i = 0; i < smokeCount; i++) {
            const eid = smokeEids[i];
            const globalMatrix = GlobalTransform.matrix.getBatch(eid);

            // Copy transform matrix directly from GlobalTransform
            const matOffset = i * 16;
            for (let j = 0; j < 16; j++) {
                transformBuffer[matOffset + j] = globalMatrix[j];
            }

            // Fill smoke data buffer: progress, seed
            const dataOffset = i * 2;
            smokeDataBuffer[dataOffset] = Progress.getProgress(eid);
            smokeDataBuffer[dataOffset + 1] = (eid * 0.137) % 1.0; // seed based on entity id
        }

        device.queue.writeBuffer(gpuShader.uniforms.projection.getGPUBuffer(device), 0, projectionMatrix as BufferSource);
        device.queue.writeBuffer(gpuShader.uniforms.transform.getGPUBuffer(device), 0, transformBuffer);
        device.queue.writeBuffer(gpuShader.uniforms.smokeData.getGPUBuffer(device), 0, smokeDataBuffer);

        renderPass.setBindGroup(0, bindGroup0);
        renderPass.setBindGroup(1, bindGroup1);
        renderPass.setPipeline(pipeline);
        renderPass.draw(6, smokeCount, 0, 0);
    };
}

