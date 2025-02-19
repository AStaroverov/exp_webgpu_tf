import { hasComponent, onAdd, onSet, query, World } from 'bitecs';
import { shaderMeta } from './sdf.shader.ts';
import { GPUShader } from '../../../WGSL/GPUShader.ts';
import { getTypeTypedArray } from '../../../Shader';
import { projectionMatrix } from '../ResizeSystem.ts';
import { Color, Roundness, Shadow } from '../../Components/Common.ts';
import { Shape } from '../../Components/Shape.ts';
import { GlobalTransform } from '../../Components/Transform.ts';

import { createChangedDetector } from '../ChangedDetectorSystem.ts';

export function createDrawShapeSystem(world: World, device: GPUDevice) {
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

    const colorChanges = createChangedDetector(world, [onAdd(Color), onSet(Color)]);
    const shapeChanges = createChangedDetector(world, [onAdd(Shape), onSet(Shape)]);
    const shadowChanges = createChangedDetector(world, [onAdd(Shadow), onSet(Shadow)]);
    const roundnessChanges = createChangedDetector(world, [onAdd(Roundness), onSet(Roundness)]);

    return function drawShapeSystem(renderPass: GPURenderPassEncoder) {
        const entities = query(world, [Shape, GlobalTransform, Color]);

        if (entities.length === 0) return;

        for (let i = 0; i < entities.length; i++) {
            const id = entities[i];

            transformCollect.set(GlobalTransform.matrix.getBatche(id), i * 16);

            // ui8
            if (shapeChanges.has(id)) {
                kindCollect[i] = Shape.kind[id];
                // vec4<f32> width, height, ..., ...
                valuesCollect.set(Shape.values.getBatche(id), i * 6);
            }

            if (roundnessChanges.has(id)) {
                // f32
                roundnessCollect[i] = Roundness.value[id];
            }

            if (colorChanges.has(id)) {
                // [f32, 4]
                colorCollect[i * 4 + 0] = Color.r[id];
                colorCollect[i * 4 + 1] = Color.g[id];
                colorCollect[i * 4 + 2] = Color.b[id];
                colorCollect[i * 4 + 3] = Color.a[id];
            }

            if (shadowChanges.has(id)) {
                // [f32, 2]
                shadowCollect[i * 2 + 0] = Shadow.fadeStart[id];
                shadowCollect[i * 2 + 1] = Shadow.fadeEnd[id];
            }

            if (hasComponent(world, id, Shadow) && shadowChanges.has(id)) {
                // [f32, 2]
                shadowCollect[i * 2 + 0] = Shadow.fadeStart[id];
                shadowCollect[i * 2 + 1] = Shadow.fadeEnd[id];
            }
        }

        device.queue.writeBuffer(gpuShader.uniforms.projection.getGPUBuffer(device), 0, projectionMatrix as Float32Array);
        device.queue.writeBuffer(gpuShader.uniforms.transform.getGPUBuffer(device), 0, transformCollect);

        if (shapeChanges.hasChanges()) {
            device.queue.writeBuffer(gpuShader.uniforms.kind.getGPUBuffer(device), 0, kindCollect);
            device.queue.writeBuffer(gpuShader.uniforms.values.getGPUBuffer(device), 0, valuesCollect);
        }

        if (colorChanges.hasChanges()) {
            device.queue.writeBuffer(gpuShader.uniforms.color.getGPUBuffer(device), 0, colorCollect);
        }
        if (shadowChanges.hasChanges()) {
            device.queue.writeBuffer(gpuShader.uniforms.shadow.getGPUBuffer(device), 0, shadowCollect);
        }
        if (roundnessChanges.hasChanges()) {
            device.queue.writeBuffer(gpuShader.uniforms.roundness.getGPUBuffer(device), 0, roundnessCollect);
        }

        renderPass.setBindGroup(0, bindGroup0);
        renderPass.setBindGroup(1, bindGroup1);

        renderPass.setPipeline(pipelineShadow);
        renderPass.draw(6, entities.length, 0, 0);

        renderPass.setPipeline(pipelineSdf);
        renderPass.draw(6, entities.length, 0, 0);
    };
}