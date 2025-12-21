import { onAdd, onSet, query, World } from 'bitecs';
import { shaderMeta } from './sdf.shader.ts';
import { GPUShader } from '../../../WGSL/GPUShader.ts';
import { getTypeTypedArray } from '../../../Shader';
import { projectionMatrix } from '../ResizeSystem.ts';
import { Color, Roundness } from '../../Components/Common.ts';
import { Shape } from '../../Components/Shape.ts';
import { GlobalTransform } from '../../Components/Transform.ts';
import { createChangeDetector } from '../ChangedDetectorSystem.ts';

export function createDrawShapeSystem({ device, world, shadowMapTexture }: {
    world: World,
    device: GPUDevice,
    shadowMapTexture: GPUTexture,
}) {
    const gpuShader = new GPUShader(shaderMeta);
    
    // Set shadow map texture on shader meta before creating bind groups
    if (shadowMapTexture) {
        shaderMeta.uniforms.shadowMap.setTexture(shadowMapTexture);
    }
    
    // Pipeline for shadow map pass (r32float, no depth, no blending)
    // Uses autoLayout with explicit bindGroups since it doesn't use all uniforms
    const pipelineShadowMap = shadowMapTexture 
        ? gpuShader.getRenderPipeline(device, 'vs_shadow_map', 'fs_shadow_map', { 
            targetFormat: 'r32float',
            autoLayout: true,
            withBlending: false,
            withDepth: false,
            bindGroups: {
                0: ['projection'],
                1: ['transform', 'kind', 'values', 'roundness'],
            },
        })
        : null;
    
    // Pipeline for visual shadow effect
    const pipelineShadow = gpuShader.getRenderPipeline(device, 'vs_shadow', 'fs_shadow', { withDepth: true });
    
    // Main shape pipeline
    const pipelineSdf = gpuShader.getRenderPipeline(device, 'vs_main', 'fs_main', { withDepth: true });
    
    // Main pass bind groups (includes shadow map in group 2)
    const bindGroup0 = gpuShader.getBindGroup(device, 0);
    const bindGroup1 = gpuShader.getBindGroup(device, 1);
    const bindGroup2 = shadowMapTexture ? gpuShader.getBindGroup(device, 2) : null;
    
    // Shadow map pass bind groups (cached during pipeline creation)
    const shadowMapBindGroup0 = pipelineShadowMap ? gpuShader.getBindGroup(device, 0, 'vs_shadow_map', 'fs_shadow_map') : null;
    const shadowMapBindGroup1 = pipelineShadowMap ? gpuShader.getBindGroup(device, 1, 'vs_shadow_map', 'fs_shadow_map') : null;

    const transformCollect = getTypeTypedArray(shaderMeta.uniforms.transform.type);
    const kindCollect = getTypeTypedArray(shaderMeta.uniforms.kind.type);
    const colorCollect = getTypeTypedArray(shaderMeta.uniforms.color.type);
    const valuesCollect = getTypeTypedArray(shaderMeta.uniforms.values.type);
    const roundnessCollect = getTypeTypedArray(shaderMeta.uniforms.roundness.type);

    const shapeChanges = createChangeDetector(world, [onAdd(Shape), onSet(Shape)]);
    const colorChanges = createChangeDetector(world, [onAdd(Color), onSet(Color)]);
    const roundnessChanges = createChangeDetector(world, [onAdd(Roundness), onSet(Roundness)]);
    let prevEntityCount = 0;
    
    function updateBuffers() {
        const entities = query(world, [GlobalTransform, Shape, Color]); // Roundness, Shadow is optional

        if (entities.length === 0) return 0;

        const countChanged = entities.length !== prevEntityCount;
        prevEntityCount = entities.length;

        for (let i = 0; i < entities.length; i++) {
            const id = entities[i];

            transformCollect.set(GlobalTransform.matrix.getBatch(id), i * 16);

            if (countChanged || shapeChanges.hasChanges()) {
                kindCollect[i] = Shape.kind[id];
                valuesCollect.set(Shape.values.getBatch(id), i * 6);
            }

            if (countChanged || colorChanges.hasChanges()) {
                colorCollect[i * 4 + 0] = Color.r[id];
                colorCollect[i * 4 + 1] = Color.g[id];
                colorCollect[i * 4 + 2] = Color.b[id];
                colorCollect[i * 4 + 3] = Color.a[id];
            }

            if (countChanged || roundnessChanges.hasChanges()) {
                roundnessCollect[i] = Roundness.value[id];
            }
        }

        device.queue.writeBuffer(gpuShader.uniforms.projection.getGPUBuffer(device), 0, projectionMatrix as BufferSource);
        device.queue.writeBuffer(gpuShader.uniforms.transform.getGPUBuffer(device), 0, transformCollect);

        if (countChanged || shapeChanges.hasChanges()) {
            device.queue.writeBuffer(gpuShader.uniforms.kind.getGPUBuffer(device), 0, kindCollect);
            device.queue.writeBuffer(gpuShader.uniforms.values.getGPUBuffer(device), 0, valuesCollect);
        }

        if (countChanged || colorChanges.hasChanges()) {
            device.queue.writeBuffer(gpuShader.uniforms.color.getGPUBuffer(device), 0, colorCollect);
        }

        if (countChanged || roundnessChanges.hasChanges()) {
            device.queue.writeBuffer(gpuShader.uniforms.roundness.getGPUBuffer(device), 0, roundnessCollect);
        }

        return entities.length;
    }

    // Main render pass - renders shapes with shadow map sampling
    function drawShapes(renderPass: GPURenderPassEncoder) {
        const entityCount = updateBuffers();
        if (entityCount === 0) return;

        renderPass.setBindGroup(0, bindGroup0);
        renderPass.setBindGroup(1, bindGroup1);
        // Set shadow map bind group (group 2) - only needed for main shapes pass
        if (bindGroup2) {
            renderPass.setBindGroup(2, bindGroup2);
            renderPass.setPipeline(pipelineShadow);
            renderPass.draw(6, entityCount, 0, 0);
        }

        // Main shapes (samples shadow map for object-to-object shadows)
        renderPass.setPipeline(pipelineSdf);
        renderPass.draw(6, entityCount, 0, 0);

        shapeChanges.clear();
        colorChanges.clear();
        roundnessChanges.clear();
    }

    // Shadow map pass - renders shadow silhouettes with Z height
    function drawShadowMap(shadowMapPass: GPURenderPassEncoder) {
        const entityCount = updateBuffers();
        if (entityCount === 0 || !pipelineShadowMap || !shadowMapBindGroup0 || !shadowMapBindGroup1) return;
        
        shadowMapPass.setPipeline(pipelineShadowMap);
        shadowMapPass.setBindGroup(0, shadowMapBindGroup0);
        shadowMapPass.setBindGroup(1, shadowMapBindGroup1);
        shadowMapPass.draw(6, entityCount, 0, 0);
    }

    return { drawShapes, drawShadowMap };
}