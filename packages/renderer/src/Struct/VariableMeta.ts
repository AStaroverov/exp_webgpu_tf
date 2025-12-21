import { getTypeBufferSize, getTypeSize } from '../Shader';

export enum VariableKind {
    Texture = 'texture',
    Sampler = 'sampler',
    Uniform = 'uniform',
    StorageRead = 'storage, read',
    StorageWrite = 'storage, write',
}

export class VariableMeta {
    public group: number;
    public binding: number;
    public visibility: GPUFlagsConstant;
    public textureSampleType?: GPUTextureSampleType;

    private size?: number;
    private bufferSize?: number;

    private sampler?: GPUSampler;
    private texture?: GPUTexture;

    constructor(
        public name: string,
        public kind: VariableKind,
        public type: string,
        private optional?: {
            group?: number,
            binding?: number,
            visibility?: GPUFlagsConstant,
            size?: number,
            bufferSize?: number,
            textureSampleType?: GPUTextureSampleType,
        },
    ) {
        this.group = optional?.group ?? 0;
        this.binding = optional?.binding ?? 0;
        this.visibility = optional?.visibility ?? (GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT);
        this.textureSampleType = optional?.textureSampleType;
    }

    public getSize(): number {
        return (this.size ??= this.optional?.size ?? getTypeSize(this.type));
    }

    public getBufferSize(): number {
        return (this.bufferSize ??= this.optional?.bufferSize ?? getTypeBufferSize(this.type));
    }

    public setTexture(texture: GPUTexture) {
        this.texture = texture;
    }

    public getTexture(): GPUTexture {
        if (!this.texture) {
            throw new Error(`Texture view for ${ this.name } is not set`);
        }
        return this.texture;
    }

    public setSampler(sampler: GPUSampler) {
        this.sampler = sampler;
    }

    public getSampler(): GPUSampler {
        if (!this.sampler) {
            throw new Error(`Sampler for ${ this.name } is not set`);
        }
        return this.sampler;
    }
}