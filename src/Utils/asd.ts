import { Variable } from '../TMP/Variable.ts';

export function getBindGroupEntry(device: GPUDevice, variableData: Variable): GPUBindGroupEntry {
    return {
        binding: variableData.variable.binding, resource: { buffer: variableData.getGPUBuffer(device) },
    };
}