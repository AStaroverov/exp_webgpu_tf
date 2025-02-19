import { VariableKind, VariableMeta } from '../Struct/VariableMeta.ts';

export class GPUVariable {
    gpuBuffer?: GPUBuffer;
    gpuGroupEntry?: GPUBindGroupEntry;
    gpuBindGroupLayoutEntry?: GPUBindGroupLayoutEntry;

    constructor(public variable: VariableMeta, public usage: GPUFlagsConstant = getUsageByKind(variable.kind)) {
    }

    getGPUBuffer(device: GPUDevice): GPUBuffer {
        return this.gpuBuffer ?? (this.gpuBuffer = device.createBuffer({
            size: this.variable.bufferSize,
            usage: this.usage,
        }));
    }

    getBindGroupEntry(device: GPUDevice): GPUBindGroupEntry {
        return this.gpuGroupEntry ?? (this.gpuGroupEntry = {
            binding: this.variable.binding,
            resource: { buffer: this.getGPUBuffer(device) },
        });
    }

    getBindGroupLayoutEntry(): GPUBindGroupLayoutEntry {
        return this.gpuBindGroupLayoutEntry ?? (this.gpuBindGroupLayoutEntry = {
            visibility: this.variable.visibility,
            binding: this.variable.binding,
            buffer: { type: mapKindToBufferType[this.variable.kind] },
        });
    }

    destroy() {
        this.gpuBuffer?.destroy();
        this.gpuBuffer = null!;
        this.gpuGroupEntry = null!;
        this.gpuBindGroupLayoutEntry = null!;
    }
}

const UNIFORM_USAGE = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
const STORAGE_USAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

function getUsageByKind(kind: VariableKind) {
    switch (kind) {
        case VariableKind.Uniform:
            return UNIFORM_USAGE;
        case VariableKind.StorageRead:
        case VariableKind.StorageWrite:
            return STORAGE_USAGE;
    }
}

const mapKindToBufferType = <const>{
    [VariableKind.Uniform]: 'uniform',
    [VariableKind.StorageRead]: 'read-only-storage',
    [VariableKind.StorageWrite]: 'storage',
};
