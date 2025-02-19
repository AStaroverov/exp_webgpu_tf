import { Changed, defineQuery, enterQuery, IWorld } from 'bitecs';

import { shaderMeta } from './rope.shader.ts';
import { GPUShader } from '../../../WGSL/GPUShader.ts';
import { getTypeTypedArray } from '../../../Shader';
import { projectionMatrix } from '../resizeSystem.ts';
import { Rope, ROPE_BUFFER_LENGTH, ROPE_POINTS_COUNT } from '../../Components/Rope.ts';
import { Color, Thinness } from '../../Components/Common.ts';
import { GlobalTransform } from '../../Components/Transform.ts';

export function createDrawRopeSystem(world: IWorld, device: GPUDevice) {
    const gpuShader = new GPUShader(shaderMeta);
    const pipeline = gpuShader.getRenderPipeline(device, 'vertex', 'fragment');
    const bindGroup0 = gpuShader.getBindGroup(device, 0);
    const bindGroup1 = gpuShader.getBindGroup(device, 1);

    const transformCollect = getTypeTypedArray(shaderMeta.uniforms.transform.type);
    const colorCollect = getTypeTypedArray(shaderMeta.uniforms.color.type);
    const pointsCollect = getTypeTypedArray(shaderMeta.uniforms.points.type);
    const thinnessCollect = getTypeTypedArray(shaderMeta.uniforms.thinness.type);

    // optimize query by each component
    const query = defineQuery([Rope, Thinness, Color, GlobalTransform]);
    const enter = enterQuery(query);
    const queryChanged = defineQuery([Changed(Rope), Changed(Thinness), Changed(Color), Changed(GlobalTransform)]);

    return function drawRopeSystem(renderPass: GPURenderPassEncoder) {
        const entities = query(world);
        const enterEntities = enter(world);
        const changedEntities = queryChanged(world);

        if (entities.length === 0) return;

        for (let i = 0; i < entities.length; i++) {
            const id = entities[i];

            if (enterEntities.indexOf(id) === -1 && changedEntities.indexOf(id) === -1) continue;

            pointsCollect.set(Rope.points[id], i * ROPE_BUFFER_LENGTH);
            transformCollect.set(GlobalTransform.matrix[id], i * 16);

            // f32
            thinnessCollect[i] = Thinness.value[id];
            // [f32, 4]
            colorCollect[i * 4 + 0] = Color.r[id];
            colorCollect[i * 4 + 1] = Color.g[id];
            colorCollect[i * 4 + 2] = Color.b[id];
            colorCollect[i * 4 + 3] = Color.a[id];
        }

        device.queue.writeBuffer(gpuShader.uniforms.projection.getGPUBuffer(device), 0, projectionMatrix as Float32Array);

        if (enterEntities.length > 0 || changedEntities.length > 0) {
            device.queue.writeBuffer(gpuShader.uniforms.color.getGPUBuffer(device), 0, colorCollect);
            device.queue.writeBuffer(gpuShader.uniforms.points.getGPUBuffer(device), 0, pointsCollect);
            device.queue.writeBuffer(gpuShader.uniforms.thinness.getGPUBuffer(device), 0, thinnessCollect);
            device.queue.writeBuffer(gpuShader.uniforms.transform.getGPUBuffer(device), 0, transformCollect);
        }

        renderPass.setPipeline(pipeline);
        renderPass.setBindGroup(0, bindGroup0);
        renderPass.setBindGroup(1, bindGroup1);
        renderPass.draw(6, entities.length * ROPE_POINTS_COUNT, 0, 0);
    };
}