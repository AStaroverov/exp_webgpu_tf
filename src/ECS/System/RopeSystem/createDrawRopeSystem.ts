import { Changed, defineQuery, enterQuery, IWorld } from 'bitecs';
import { Rope, ROPE_BUFFER_LENGTH, ROPE_SEGMENTS_COUNT } from '../../Component/Rope.ts';
import { shaderMeta } from './rope.shader.ts';
import { GPUShader } from '../../../WGSL/GPUShader.ts';
import { getTypeTypedArray } from '../../../Shader';
import { Color, Resolution, Thinness, Translate } from '../../Component/Common.ts';

export function createDrawRopeSystem(world: IWorld, device: GPUDevice) {
    const gpuShader = new GPUShader(shaderMeta);
    const pipeline = gpuShader.getRenderPipeline(device);
    const bindGroup = gpuShader.getBindGroup(device, 0);

    const colorCollect = getTypeTypedArray(shaderMeta.uniforms.color.type);
    const pointsCollect = getTypeTypedArray(shaderMeta.uniforms.points.type);
    const thinnessCollect = getTypeTypedArray(shaderMeta.uniforms.thinness.type);
    const translateCollect = getTypeTypedArray(shaderMeta.uniforms.translate.type);
    const resolutionCollect = getTypeTypedArray(shaderMeta.uniforms.resolution.type);

    const query = defineQuery([Rope, Thinness, Color, Translate, Resolution]);
    const enter = enterQuery(query);
    const queryChanged = defineQuery([Changed(Rope), Changed(Thinness), Changed(Color), Changed(Translate), Changed(Resolution)]);

    return function drawRopeSystem(renderPass: GPURenderPassEncoder) {
        const entities = query(world);
        const enterEntities = enter(world);
        const changedEntities = queryChanged(world);

        if (entities.length === 0) return;

        for (let i = 0; i < entities.length; i++) {
            const id = entities[i];

            if (enterEntities.indexOf(id) === -1 && changedEntities.indexOf(id) === -1) continue;

            pointsCollect.set(Rope.points[id], i * ROPE_BUFFER_LENGTH);
            // vec2<f32>
            translateCollect[i * 2] = Translate.x[id];
            translateCollect[i * 2 + 1] = Translate.y[id];
            // vec2<f32>
            resolutionCollect[i * 2] = Resolution.x[id];
            resolutionCollect[i * 2 + 1] = Resolution.y[id];

            // f32
            thinnessCollect[i] = Thinness.value[id];
            // [f32, 4]
            colorCollect[i * 4 + 0] = Color.r[id];
            colorCollect[i * 4 + 1] = Color.g[id];
            colorCollect[i * 4 + 2] = Color.b[id];
            colorCollect[i * 4 + 3] = Color.a[id];
        }

        if (enterEntities.length > 0 || changedEntities.length > 0) {
            device.queue.writeBuffer(gpuShader.uniforms.color.getGPUBuffer(device), 0, colorCollect);
            device.queue.writeBuffer(gpuShader.uniforms.points.getGPUBuffer(device), 0, pointsCollect);
            device.queue.writeBuffer(gpuShader.uniforms.thinness.getGPUBuffer(device), 0, thinnessCollect);
            device.queue.writeBuffer(gpuShader.uniforms.translate.getGPUBuffer(device), 0, translateCollect);
            device.queue.writeBuffer(gpuShader.uniforms.resolution.getGPUBuffer(device), 0, resolutionCollect);
        }

        renderPass.setPipeline(pipeline);
        renderPass.setBindGroup(0, bindGroup);
        renderPass.draw(6, entities.length * ROPE_SEGMENTS_COUNT, 0, 0);
        renderPass.end();
    };
}