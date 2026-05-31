import { GPUShader } from 'renderer/src/WGSL/GPUShader.ts';
import { getTypeTypedArray } from 'renderer/src/Shader/index.ts';
import { projectionMatrix } from 'renderer/src/ECS/Systems/ResizeSystem.ts';
import { RenderDI } from '../../../../DI/RenderDI.ts';
import { MapDI } from '../../../../DI/MapDI.ts';
import { HexGridConfig } from '../../../../Map/HexConfig.ts';
import { shaderMeta, MAX_HEX_COUNT } from './grid.shader.ts';

const SQRT3_2 = Math.sqrt(3) / 2;

// Line/fill appearance of the debug grid.
const LINE_COLOR: [number, number, number, number] = [0.12, 0.09, 0.05, 0.55];
const LINE_WIDTH = 3; // world units
const FILL_ALPHA = 0.04;

export function createDrawGridSystem({ device } = RenderDI, { grid } = MapDI) {
    const gpuShader = new GPUShader(shaderMeta);
    const pipeline = gpuShader.getRenderPipeline(device, 'vs_main', 'fs_main', { withDepth: true });
    const bindGroup0 = gpuShader.getBindGroup(device, 0);
    const bindGroup1 = gpuShader.getBindGroup(device, 1);

    // The grid is static — build the per-cell center buffer once.
    const cellsBuffer = getTypeTypedArray(shaderMeta.uniforms.cells.type);
    let count = 0;
    grid.forEachCell((_, hex) => {
        if (count >= MAX_HEX_COUNT) return;
        const o = count * 4;
        cellsBuffer[o] = hex.x + grid.originX;
        cellsBuffer[o + 1] = hex.y + grid.originY;
        count++;
    });

    const params = new Float32Array([
        HexGridConfig.radius, // circumradius (quad half-size)
        HexGridConfig.radius * SQRT3_2, // inradius (half-width)
        LINE_WIDTH,
        FILL_ALPHA,
    ]);
    const color = new Float32Array(LINE_COLOR);

    let uploaded = false;

    return (renderPass: GPURenderPassEncoder) => {
        if (count === 0) return;

        device.queue.writeBuffer(gpuShader.uniforms.projection.getGPUBuffer(device), 0, projectionMatrix as BufferSource);

        if (!uploaded) {
            device.queue.writeBuffer(gpuShader.uniforms.params.getGPUBuffer(device), 0, params);
            device.queue.writeBuffer(gpuShader.uniforms.color.getGPUBuffer(device), 0, color);
            device.queue.writeBuffer(gpuShader.uniforms.cells.getGPUBuffer(device), 0, cellsBuffer);
            uploaded = true;
        }

        renderPass.setBindGroup(0, bindGroup0);
        renderPass.setBindGroup(1, bindGroup1);
        renderPass.setPipeline(pipeline);
        renderPass.draw(6, count, 0, 0);
    };
}
