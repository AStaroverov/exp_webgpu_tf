import {VariableMeta} from "../Struct/VariableMeta.ts";
import {getTypeConstructor} from "../Shader";

export class Variable {
    array: Float32Array | Uint32Array | Int32Array;
    buffer: ArrayBuffer;
    gpuBuffer?: GPUBuffer;
    gpuGroupEntry?: GPUBindGroupEntry;

    constructor(public variable: VariableMeta, public usage: GPUFlagsConstant) {
        this.buffer = new ArrayBuffer(variable.bufferSize);
        this.array = new (getTypeConstructor(variable.type))(this.buffer);
    }

    set(value: number[], offset = 0) {
        this.array.set(value, offset);
    }

    getArray() {
        return this.array;
    }

    getBuffer() {
        return this.array.buffer;
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
        this.array = null!;
    }
}
