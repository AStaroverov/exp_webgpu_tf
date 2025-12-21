import { VariableKind, VariableMeta } from '../Struct/VariableMeta.ts';

export class GPUVariable {
    gpuBuffer?: GPUBuffer;
    gpuGroupEntry?: GPUBindGroupEntry;
    gpuBindGroupLayoutEntry?: GPUBindGroupLayoutEntry;

    constructor(public variable: VariableMeta, public usage: GPUFlagsConstant = getUsageByKind(variable.kind)) {
    }

    getGPUBuffer(device: GPUDevice): GPUBuffer {
        return this.gpuBuffer ?? (this.gpuBuffer = device.createBuffer({
            size: this.variable.getBufferSize(),
            usage: this.usage,
        }));
    }

    getGPUResource(device: GPUDevice): GPUBindingResource {
        if (this.variable.type.startsWith('texture')) {
            return this.variable.getTexture().createView();
        }
        if (this.variable.type === 'sampler') {
            return this.variable.getSampler();
        }
        return { buffer: this.getGPUBuffer(device) };
    }

    getBindGroupEntry(device: GPUDevice): GPUBindGroupEntry {
        return this.gpuGroupEntry ?? (this.gpuGroupEntry = {
            binding: this.variable.binding,
            resource: this.getGPUResource(device),
        });
    }

    getBindGroupLayoutEntrySpecific(): Pick<GPUBindGroupLayoutEntry, 'buffer' | 'texture' | 'sampler'> {
        if (this.variable.kind === VariableKind.Texture) {
            return { 
                texture: this.variable.textureSampleType 
                    ? { sampleType: this.variable.textureSampleType } 
                    : {} 
            };
        }
        if (this.variable.kind === VariableKind.Sampler) {
            return { sampler: {} };
        }
        return { buffer: { type: mapKindToBufferType[this.variable.kind] } };
    }

    getBindGroupLayoutEntry(): GPUBindGroupLayoutEntry {
        return this.gpuBindGroupLayoutEntry ?? (this.gpuBindGroupLayoutEntry = {
            visibility: this.variable.visibility,
            binding: this.variable.binding,
            ...this.getBindGroupLayoutEntrySpecific(),
        });
    }

    destroy() {
        this.gpuBuffer?.destroy();
        this.gpuBuffer = null!;
        this.gpuGroupEntry = null!;
        this.gpuBindGroupLayoutEntry = null!;
    }
}

const TEXTURE_USAGE = GPUTextureUsage.TEXTURE_BINDING;
const UNIFORM_USAGE = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
const STORAGE_USAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

function getUsageByKind(kind: VariableKind) {
    switch (kind) {
        case VariableKind.Sampler:
        case VariableKind.Texture:
            return TEXTURE_USAGE;
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
