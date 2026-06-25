import { VariableKind, VariableMeta } from "../Struct/VariableMeta.ts";

export class GPUVariable {
  gpuBuffer?: GPUBuffer;
  gpuGroupEntry?: GPUBindGroupEntry;
  gpuBindGroupLayoutEntry?: GPUBindGroupLayoutEntry;

  constructor(
    public variable: VariableMeta,
    public usage: GPUFlagsConstant = getUsageByKind(variable.kind),
  ) {}

  getGPUBuffer(device: GPUDevice): GPUBuffer {
    return (
      this.gpuBuffer ??
      (this.gpuBuffer = device.createBuffer({
        size: this.variable.getBufferSize(),
        usage: this.usage,
      }))
    );
  }

  getGPUResource(device: GPUDevice): GPUBindingResource {
    if (this.variable.type.startsWith("texture")) {
      // viewDimension defaults to "2d"; for a 2D-array binding the declared
      // dimension MUST be passed so the view spans all array layers.
      return this.variable.viewDimension
        ? this.variable.getTexture().createView({ dimension: this.variable.viewDimension })
        : this.variable.getTexture().createView();
    }
    if (this.variable.type === "sampler") {
      return this.variable.getSampler();
    }
    return { buffer: this.getGPUBuffer(device) };
  }

  getBindGroupEntry(device: GPUDevice): GPUBindGroupEntry {
    return (
      this.gpuGroupEntry ??
      (this.gpuGroupEntry = {
        binding: this.variable.binding,
        resource: this.getGPUResource(device),
      })
    );
  }

  getBindGroupLayoutEntrySpecific(): Pick<
    GPUBindGroupLayoutEntry,
    "buffer" | "texture" | "sampler" | "storageTexture"
  > {
    if (this.variable.kind === VariableKind.Texture) {
      const texture: GPUTextureBindingLayout = {};
      if (this.variable.textureSampleType) texture.sampleType = this.variable.textureSampleType;
      if (this.variable.viewDimension) texture.viewDimension = this.variable.viewDimension;
      return { texture };
    }
    if (this.variable.kind === VariableKind.StorageTexture) {
      const storageTexture: GPUStorageTextureBindingLayout = {
        access: this.variable.storageTextureAccess,
        format: this.variable.storageTextureFormat!,
      };
      if (this.variable.viewDimension) storageTexture.viewDimension = this.variable.viewDimension;
      return { storageTexture };
    }
    if (this.variable.kind === VariableKind.Sampler) {
      return { sampler: {} };
    }
    return { buffer: { type: mapKindToBufferType[this.variable.kind] } };
  }

  getBindGroupLayoutEntry(): GPUBindGroupLayoutEntry {
    return (
      this.gpuBindGroupLayoutEntry ??
      (this.gpuBindGroupLayoutEntry = {
        visibility: this.variable.visibility,
        binding: this.variable.binding,
        ...this.getBindGroupLayoutEntrySpecific(),
      })
    );
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
// COPY_SRC so compute outputs (surfel buffers, etc.) can be read back to the CPU
// for diagnostics; harmless for read-only storage too.
const STORAGE_USAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

function getUsageByKind(kind: VariableKind) {
  switch (kind) {
    case VariableKind.Sampler:
    case VariableKind.Texture:
    // StorageTexture has no GPUBuffer; the GPUTexture is created externally with its
    // own usage (STORAGE_BINDING|TEXTURE_BINDING) and set via setTexture. This value
    // is unused, but the switch must stay exhaustive.
    case VariableKind.StorageTexture:
      return TEXTURE_USAGE;
    case VariableKind.Uniform:
      return UNIFORM_USAGE;
    case VariableKind.StorageRead:
    case VariableKind.StorageWrite:
      return STORAGE_USAGE;
  }
}

const mapKindToBufferType = <const>{
  [VariableKind.Uniform]: "uniform",
  [VariableKind.StorageRead]: "read-only-storage",
  [VariableKind.StorageWrite]: "storage",
};
