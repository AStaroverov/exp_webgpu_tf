import { getTypeBufferSize, getTypeSize } from "../Shader";

export enum VariableKind {
  Texture = "texture",
  Sampler = "sampler",
  Uniform = "uniform",
  StorageRead = "storage, read",
  StorageWrite = "storage, write",
  // Writable storage texture (texture_storage_Nd<format, access>). The compute pass
  // writes voxels via textureStore; later passes read the SAME GPUTexture as a sampled
  // texture_Nd<f32>. Carries `storageTextureFormat` + `storageTextureAccess`.
  StorageTexture = "storage_texture",
}

export class VariableMeta {
  public group: number;
  public binding: number;
  public visibility: GPUFlagsConstant;
  public textureSampleType?: GPUTextureSampleType;
  public viewDimension?: GPUTextureViewDimension;
  // StorageTexture only: WGSL/layout format + access (default "write-only").
  public storageTextureFormat?: GPUTextureFormat;
  public storageTextureAccess: GPUStorageTextureAccess;

  private size?: number;
  private bufferSize?: number;

  private sampler?: GPUSampler;
  private texture?: GPUTexture;

  constructor(
    public name: string,
    public kind: VariableKind,
    public type: string,
    private optional?: {
      group?: number;
      binding?: number;
      visibility?: GPUFlagsConstant;
      size?: number;
      bufferSize?: number;
      textureSampleType?: GPUTextureSampleType;
      viewDimension?: GPUTextureViewDimension;
      storageTextureFormat?: GPUTextureFormat;
      storageTextureAccess?: GPUStorageTextureAccess;
    },
  ) {
    this.group = optional?.group ?? 0;
    this.binding = optional?.binding ?? 0;
    this.visibility = optional?.visibility ?? GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
    this.textureSampleType = optional?.textureSampleType;
    this.viewDimension = optional?.viewDimension;
    this.storageTextureFormat = optional?.storageTextureFormat;
    this.storageTextureAccess = optional?.storageTextureAccess ?? "write-only";
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
      throw new Error(`Texture view for ${this.name} is not set`);
    }
    return this.texture;
  }

  public setSampler(sampler: GPUSampler) {
    this.sampler = sampler;
  }

  public getSampler(): GPUSampler {
    if (!this.sampler) {
      throw new Error(`Sampler for ${this.name} is not set`);
    }
    return this.sampler;
  }
}
