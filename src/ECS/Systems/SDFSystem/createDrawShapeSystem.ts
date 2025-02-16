import { Changed, defineQuery, enterQuery, IWorld } from 'bitecs';
import { shaderMeta } from './sdf.shader.ts';
import { GPUShader } from '../../../WGSL/GPUShader.ts';
import { getTypeTypedArray } from '../../../Shader';
import { projectionMatrix } from '../resizeSystem.ts';
import { Color, Roundness, Shadow } from '../../Components/Common.ts';
import { Shape } from '../../Components/Shape.ts';
import { GlobalTransform } from '../../Components/Transform.ts';

export function createDrawShapeSystem(world: IWorld, device: GPUDevice) {
    const gpuShader = new GPUShader(shaderMeta);
    const pipelineSdf = gpuShader.getRenderPipeline(device, 'vs_main', 'fs_main');
    const pipelineShadow = gpuShader.getRenderPipeline(device, 'vs_shadow', 'fs_shadow');
    const bindGroup0 = gpuShader.getBindGroup(device, 0);
    const bindGroup1 = gpuShader.getBindGroup(device, 1);

    const transformCollect = getTypeTypedArray(shaderMeta.uniforms.transform.type);
    const kindCollect = getTypeTypedArray(shaderMeta.uniforms.kind.type);
    const colorCollect = getTypeTypedArray(shaderMeta.uniforms.color.type);
    const valuesCollect = getTypeTypedArray(shaderMeta.uniforms.values.type);
    const roundnessCollect = getTypeTypedArray(shaderMeta.uniforms.roundness.type);
    const shadowCollect = getTypeTypedArray(shaderMeta.uniforms.shadow.type);

    const query = defineQuery([Shape, GlobalTransform, Color, Shadow]);
    const enter = enterQuery(query);
    const queryChanged = defineQuery([Changed(Shape), Changed(GlobalTransform), Changed(Color)]);
    const queryShadowChanged = defineQuery([Shape, Changed(Shadow)]);

    return function drawShapeSystem(renderPass: GPURenderPassEncoder) {
        const entities = query(world);
        const enterEntities = enter(world);
        const changedEntities = queryChanged(world);
        const shadowChangedEntities = queryShadowChanged(world);

        if (entities.length === 0) return;

        for (let i = 0; i < entities.length; i++) {
            const id = entities[i];
            const isNew = enterEntities.indexOf(id) === -1;

            if (isNew || changedEntities.indexOf(id) === -1) {
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

            if (isNew || shadowChangedEntities.indexOf(id) === -1) {
                // [f32, 2]
                shadowCollect[i * 2 + 0] = Shadow.fadeStart[id];
                shadowCollect[i * 2 + 1] = Shadow.fadeEnd[id];
            }
        }

        device.queue.writeBuffer(gpuShader.uniforms.projection.getGPUBuffer(device), 0, projectionMatrix as Float32Array);

        if (enterEntities.length > 0 || changedEntities.length > 0) {
            device.queue.writeBuffer(gpuShader.uniforms.kind.getGPUBuffer(device), 0, kindCollect);
            device.queue.writeBuffer(gpuShader.uniforms.color.getGPUBuffer(device), 0, colorCollect);
            device.queue.writeBuffer(gpuShader.uniforms.values.getGPUBuffer(device), 0, valuesCollect);
            device.queue.writeBuffer(gpuShader.uniforms.roundness.getGPUBuffer(device), 0, roundnessCollect);
            device.queue.writeBuffer(gpuShader.uniforms.transform.getGPUBuffer(device), 0, transformCollect);
        }

        if (enterEntities.length > 0 || shadowChangedEntities.length > 0) {
            device.queue.writeBuffer(gpuShader.uniforms.shadow.getGPUBuffer(device), 0, shadowCollect);
        }

        renderPass.setBindGroup(0, bindGroup0);
        renderPass.setBindGroup(1, bindGroup1);

        renderPass.setPipeline(pipelineShadow);
        renderPass.draw(6, entities.length, 0, 0);

        renderPass.setPipeline(pipelineSdf);
        renderPass.draw(6, entities.length, 0, 0);
    };
}