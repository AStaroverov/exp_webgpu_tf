import { getTypeBufferSize, getTypeSize } from '../Shader';

export enum VariableKind {
    Uniform = 'uniform',
    StorageRead = 'storage, read',
    StorageWrite = 'storage, write',
}

export class VariableMeta {
    constructor(
        public name: string,
        public kind: VariableKind,
        public type: string,
        public group: number = 0,
        public binding: number = 0,
        public visibility = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        public size = getTypeSize(type),
        public bufferSize = getTypeBufferSize(type),
    ) {
    }
}