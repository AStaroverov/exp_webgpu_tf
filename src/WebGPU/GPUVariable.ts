import {VariableKind, VariableMeta} from "../Struct/VariableMeta.ts";

export class GPUVariable {
    gpuBuffer?: GPUBuffer;
    gpuGroupEntry?: GPUBindGroupEntry;

    constructor(public variable: VariableMeta, public usage: GPUFlagsConstant = getUsageByKind(variable.kind)) {
    }

    getGPUBuffer(device: GPUDevice) {
        return this.gpuBuffer ?? (this.gpuBuffer = device.createBuffer({
            size: this.variable.bufferSize,
            usage: this.usage,
        }));
    }

    getBindGroupEntry(device: GPUDevice): GPUBindGroupEntry {
        return this.gpuGroupEntry ?? (this.gpuGroupEntry = {
            binding: this.variable.binding, resource: {buffer: this.getGPUBuffer(device)}
        })
    }

    destroy() {
        this.gpuGroupEntry = null!;
        this.gpuBuffer?.destroy();
        this.gpuBuffer = null!;
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