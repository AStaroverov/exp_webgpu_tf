// Iso voxel-radiance MIP PYRAMID sub-system. Owns the downsample shader/pipeline + the per-level
// bind groups/buffers, and issues the mip-build pass. Extracted verbatim from createVoxelSystem:
// the module builds its per-level state in rebindGrid() (called after voxelRadiance is recreated)
// and runs one compute dispatch per level in run(). Behavior is byte-for-byte identical.
import { GPUShader } from "../../../../../WGSL/GPUShader.ts";
import { getTypeTypedArray } from "../../../../../Shader/index.ts";
import { shaderMeta as mipMeta, WORKGROUP as MIP_WG } from "./voxelMip.shader.ts";
import { runMips } from "../../core/mipPass.ts";

export type MipPyramidDeps = {
  device: GPUDevice;
};

export function createMipPyramidSystem(deps: MipPyramidDeps) {
  const { device } = deps;

  // Voxel-radiance mip-pyramid downsample (one dispatch per level).
  const mipShader = new GPUShader(mipMeta);
  const mipPipeline = mipShader.getComputePipeline(device, "main");
  // The mip shader uses groups 0 (uniform+src) and 2 (dst storage) with NOTHING in group 1,
  // so its pipeline layout has an empty layout at index 1. Bind a matching empty group there
  // each dispatch, so strict implementations that require every layout index to be set are
  // satisfied. (The layout object is the same one the pipeline layout reflects.)
  const mipEmptyGroup1 = device.createBindGroup({
    layout: mipShader.createBindGroupLayout(device, 1),
    entries: [],
  });

  // Mip scratch: .xyz = destination mip dims (re-uploaded per level).
  const mipArr = getTypeTypedArray(mipMeta.uniforms.mip.type); // Int32Array(4)

  // Mip-pyramid state (rebuilt by rebindGrid). One downsample step per pair of adjacent
  // levels → mipCount-1 steps, indexed 0..mipCount-2. mipGroup0[L]/mipGroup2[L] downsample
  // mip L → mip L+1.
  let mipCount = 1;
  let mipBuf: GPUBuffer[] = [];
  let mipGroup0: GPUBindGroup[] = [];
  let mipGroup2: GPUBindGroup[] = [];
  // Grid dims of the current voxelRadiance (drive the per-level dispatch counts).
  let dimX = 1;
  let dimY = 1;
  let dimZ = 1;

  // Rebuild the mip-pyramid downsample groups (mip L → L+1) for a freshly (re)created voxelRadiance
  // texture. srcView is a single-mip SAMPLED view of mip L; dstView is a single-mip STORAGE view of
  // mip L+1 (different subresources of the same texture → allowed). Called from buildGrid after
  // voxelRadiance is recreated (views/buffers depend on the new dims).
  function rebindGrid(
    voxelRadiance: GPUTexture,
    newDimX: number,
    newDimY: number,
    newDimZ: number,
    newMipCount: number,
  ) {
    dimX = newDimX;
    dimY = newDimY;
    dimZ = newDimZ;
    for (const b of mipBuf) b.destroy();
    mipCount = newMipCount;
    mipBuf = [];
    mipGroup0 = [];
    mipGroup2 = [];
    for (let L = 0; L < mipCount - 1; L++) {
      const buf = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      mipBuf.push(buf);

      const srcView = voxelRadiance.createView({
        dimension: "3d",
        baseMipLevel: L,
        mipLevelCount: 1,
      });
      const dstView = voxelRadiance.createView({
        dimension: "3d",
        baseMipLevel: L + 1,
        mipLevelCount: 1,
      });

      mipGroup0.push(
        device.createBindGroup({
          layout: mipPipeline.getBindGroupLayout(0),
          entries: [
            { binding: mipMeta.uniforms.mip.binding, resource: { buffer: buf } },
            { binding: mipMeta.uniforms.src.binding, resource: srcView },
          ],
        }),
      );
      mipGroup2.push(
        device.createBindGroup({
          layout: mipPipeline.getBindGroupLayout(2),
          entries: [{ binding: mipMeta.uniforms.dst.binding, resource: dstView }],
        }),
      );

      // Destination dims = mip L+1 dims (halved per axis, floored at 1).
      mipArr[0] = Math.max(1, dimX >> (L + 1));
      mipArr[1] = Math.max(1, dimY >> (L + 1));
      mipArr[2] = Math.max(1, dimZ >> (L + 1));
      mipArr[3] = 0;
      device.queue.writeBuffer(buf, 0, mipArr);
    }
  }

  // Build the voxelRadiance mip pyramid: one compute pass PER level (passes are ordered/
  // barriered by the encoder, so level L+1 sees level L's writes — dispatches WITHIN a pass
  // are not synchronized, hence one pass each). Must run AFTER voxelize() (level 0 reads
  // mip 0 that voxelize wrote) and in the SAME encoder.
  function run(encoder: GPUCommandEncoder) {
    runMips(encoder, {
      pipeline: mipPipeline,
      emptyGroup1: mipEmptyGroup1,
      group0: mipGroup0,
      group2: mipGroup2,
      count: mipCount,
      dimX,
      dimY,
      dimZ,
      workgroup: MIP_WG,
      label: "mips",
    });
  }

  return {
    run,
    rebindGrid,
    get mipCount() {
      return mipCount;
    },
  };
}
