import { Changed, defineQuery, enterQuery, IWorld } from 'bitecs';
import { shaderMeta } from './sdf.shader.ts';
import { GPUShader } from '../../../WGSL/GPUShader.ts';
import { getTypeTypedArray } from '../../../Shader';
import { Color, Resolution, Roundness, Thinness, Translate } from '../../Component/Common.ts';
import { Shape } from '../../Component/Shape.ts';

export function createDrawShapeSystem(world: IWorld, device: GPUDevice) {
    const gpuShader = new GPUShader(shaderMeta);
    const pipeline = gpuShader.getRenderPipeline(device);
    const bindGroup = gpuShader.getBindGroup(device);

    const kindCollect = getTypeTypedArray(shaderMeta.uniforms.kind.type);
    const colorCollect = getTypeTypedArray(shaderMeta.uniforms.color.type);
    const point1Collect = getTypeTypedArray(shaderMeta.uniforms.point1.type);
    const point2Collect = getTypeTypedArray(shaderMeta.uniforms.point2.type);
    const thinnessCollect = getTypeTypedArray(shaderMeta.uniforms.thinness.type);
    const roundnessCollect = getTypeTypedArray(shaderMeta.uniforms.roundness.type);
    const translateCollect = getTypeTypedArray(shaderMeta.uniforms.translate.type);
    const resolutionCollect = getTypeTypedArray(shaderMeta.uniforms.resolution.type);

    const query = defineQuery([Shape, Thinness, Color, Translate, Resolution]);
    const enter = enterQuery(query);
    const queryChanged = defineQuery([Changed(Shape), Changed(Thinness), Changed(Color), Changed(Translate), Changed(Resolution)]);

    return function drawShapeSystem(renderPass: GPURenderPassEncoder) {
        const entities = query(world);
        const enterEntities = enter(world);
        const changedEntities = queryChanged(world);

        if (entities.length === 0) return;

        for (let i = 0; i < entities.length; i++) {
            const id = entities[i];

            if (enterEntities.indexOf(id) === -1 && changedEntities.indexOf(id) === -1) continue;

            // vec2<f32>
            translateCollect[i * 2] = Translate.x[id];
            translateCollect[i * 2 + 1] = Translate.y[id];
            // vec2<f32>
            resolutionCollect[i * 2] = Resolution.x[id];
            resolutionCollect[i * 2 + 1] = Resolution.y[id];

            // ui8
            kindCollect[i] = Shape.kind[id];

            // vec2<f32>
            point1Collect[i * 2] = Shape.point1[id][0];
            point1Collect[i * 2 + 1] = Shape.point1[id][1];

            // vec2<f32>
            point2Collect[i * 2] = Shape.point2[id][0];
            point2Collect[i * 2 + 1] = Shape.point2[id][1];

            // f32
            thinnessCollect[i] = Thinness.value[id];
            // f32
            roundnessCollect[i] = Roundness.value[id];
            // [f32, 4]
            colorCollect[i * 4 + 0] = Color.r[id];
            colorCollect[i * 4 + 1] = Color.g[id];
            colorCollect[i * 4 + 2] = Color.b[id];
            colorCollect[i * 4 + 3] = Color.a[id];
        }

        if (enterEntities.length > 0 || changedEntities.length > 0) {
            device.queue.writeBuffer(gpuShader.uniforms.kind.getGPUBuffer(device), 0, kindCollect);
            device.queue.writeBuffer(gpuShader.uniforms.color.getGPUBuffer(device), 0, colorCollect);
            device.queue.writeBuffer(gpuShader.uniforms.point1.getGPUBuffer(device), 0, point1Collect);
            device.queue.writeBuffer(gpuShader.uniforms.point2.getGPUBuffer(device), 0, point2Collect);
            device.queue.writeBuffer(gpuShader.uniforms.thinness.getGPUBuffer(device), 0, thinnessCollect);
            device.queue.writeBuffer(gpuShader.uniforms.roundness.getGPUBuffer(device), 0, roundnessCollect);
            device.queue.writeBuffer(gpuShader.uniforms.translate.getGPUBuffer(device), 0, translateCollect);
            device.queue.writeBuffer(gpuShader.uniforms.resolution.getGPUBuffer(device), 0, resolutionCollect);
        }

        renderPass.setPipeline(pipeline);
        renderPass.setBindGroup(0, bindGroup);
        renderPass.draw(6, entities.length, 0, 0);
        renderPass.end();
    };
}