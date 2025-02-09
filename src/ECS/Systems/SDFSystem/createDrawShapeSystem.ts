import { Changed, defineQuery, enterQuery, IWorld } from 'bitecs';
import { shaderMeta } from './sdf.shader.ts';
import { GPUShader } from '../../../WGSL/GPUShader.ts';
import { getTypeTypedArray } from '../../../Shader';
import { projectionMatrix } from '../resizeSystem.ts';
import { Color, Roundness } from '../../Components/Common.ts';
import { Shape } from '../../Components/Shape.ts';
import { GlobalTransform } from '../../Components/Transform.ts';

export function createDrawShapeSystem(world: IWorld, device: GPUDevice) {
    const gpuShader = new GPUShader(shaderMeta);
    const pipeline = gpuShader.getRenderPipeline(device);
    const bindGroup0 = gpuShader.getBindGroup(device, 0);
    const bindGroup1 = gpuShader.getBindGroup(device, 1);

    const transformCollect = getTypeTypedArray(shaderMeta.uniforms.transform.type);
    const kindCollect = getTypeTypedArray(shaderMeta.uniforms.kind.type);
    const colorCollect = getTypeTypedArray(shaderMeta.uniforms.color.type);
    const valuesCollect = getTypeTypedArray(shaderMeta.uniforms.values.type);
    const roundnessCollect = getTypeTypedArray(shaderMeta.uniforms.roundness.type);

    const query = defineQuery([Shape, Color, GlobalTransform]);
    const enter = enterQuery(query);
    const queryChanged = defineQuery([Changed(Shape), Changed(Color), Changed(GlobalTransform)]);

    return function drawShapeSystem(renderPass: GPURenderPassEncoder) {
        const entities = query(world);
        const enterEntities = enter(world);
        const changedEntities = queryChanged(world);

        if (entities.length === 0) return;

        for (let i = 0; i < entities.length; i++) {
            const id = entities[i];

            if (enterEntities.indexOf(id) === -1 && changedEntities.indexOf(id) === -1) continue;

            transformCollect.set(GlobalTransform.matrix[id], i * 16);
            // ui8
            kindCollect[i] = Shape.kind[id];
            // vec4<f32> width, height, ..., ...
            valuesCollect.set(Shape.values[id], i * 6);
            // f32
            roundnessCollect[i] = Roundness.value[id];
            // [f32, 4]
            colorCollect[i * 4 + 0] = Color.r[id];
            colorCollect[i * 4 + 1] = Color.g[id];
            colorCollect[i * 4 + 2] = Color.b[id];
            colorCollect[i * 4 + 3] = Color.a[id];
        }

        device.queue.writeBuffer(gpuShader.uniforms.projection.getGPUBuffer(device), 0, projectionMatrix as Float32Array);

        if (enterEntities.length > 0 || changedEntities.length > 0) {
            device.queue.writeBuffer(gpuShader.uniforms.kind.getGPUBuffer(device), 0, kindCollect);
            device.queue.writeBuffer(gpuShader.uniforms.color.getGPUBuffer(device), 0, colorCollect);
            device.queue.writeBuffer(gpuShader.uniforms.values.getGPUBuffer(device), 0, valuesCollect);
            device.queue.writeBuffer(gpuShader.uniforms.roundness.getGPUBuffer(device), 0, roundnessCollect);
            device.queue.writeBuffer(gpuShader.uniforms.transform.getGPUBuffer(device), 0, transformCollect);
        }

        renderPass.setPipeline(pipeline);
        renderPass.setBindGroup(0, bindGroup0);
        renderPass.setBindGroup(1, bindGroup1);
        renderPass.draw(6, entities.length, 0, 0);
    };
}