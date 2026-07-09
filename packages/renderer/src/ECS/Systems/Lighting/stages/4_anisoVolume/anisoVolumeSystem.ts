// Anisotropic directional-pyramid sub-system (the far-field anti-leak). Owns the BASE + VOLUME
// downsample shaders/pipelines, the 6 directional volumes (AnisoTextures) + their per-level bind
// groups/buffers, and issues the two build passes. Extracted verbatim from createVoxelSystem:
// rebindGrid() rebuilds the per-grid state (called after voxelRadiance is recreated, from buildGrid)
// and base()/mips() issue the dispatches. Behavior is byte-for-byte identical.
//
// The BASE pass builds the 6 directional level-0 volumes from iso voxelRadiance mip 0; the VOLUME
// pass downsamples each direction one level per dispatch. Both use groups 0 (uniform + sources) and
// 2 (storage outputs) with an EMPTY group 1 (same shape as the mip pass → bind a matching empty
// group so strict impls are satisfied).
import { GPUShader } from "../../../../../WGSL/GPUShader.ts";
import { getTypeTypedArray } from "../../../../../Shader/index.ts";
import { shaderMeta as anisoBaseMeta, WORKGROUP as ANISO_WG } from "./voxelAnisoBase.shader.ts";
import { shaderMeta as anisoVolMeta } from "./voxelAnisoVolume.shader.ts";
import {
  anisoBaseDims,
  createAnisoTextures,
  voxelMipLevelCount,
  type AnisoTextures,
} from "../../core/voxelResources.ts";
import { runAnisoBase, runAnisoMips } from "../../core/mipPass.ts";

export type AnisoVolumeDeps = {
  device: GPUDevice;
};

// Grid box for the aniso pyramid rebuild (mirrors the fields buildGrid passes for createVoxelTextures).
export type AnisoGridBox = {
  originX: number;
  originY: number;
  originZ: number;
  dimX: number;
  dimY: number;
  dimZ: number;
  cellSize: number;
};

export function createAnisoVolumeSystem(deps: AnisoVolumeDeps) {
  const { device } = deps;

  // Anisotropic directional pyramid (the far-field anti-leak). BASE builds the 6 directional
  // level-0 volumes from iso voxelRadiance mip 0; VOLUME downsamples each direction one level per
  // dispatch. Both use groups 0 (uniform + sources) and 2 (storage outputs) with an EMPTY group 1
  // (same shape as the mip pass → bind a matching empty group so strict impls are satisfied).
  const anisoBaseShader = new GPUShader(anisoBaseMeta);
  const anisoBasePipeline = anisoBaseShader.getComputePipeline(device, "main");
  const anisoBaseEmptyGroup1 = device.createBindGroup({
    layout: anisoBaseShader.createBindGroupLayout(device, 1),
    entries: [],
  });
  const anisoVolShader = new GPUShader(anisoVolMeta);
  const anisoVolPipeline = anisoVolShader.getComputePipeline(device, "main");
  const anisoVolEmptyGroup1 = device.createBindGroup({
    layout: anisoVolShader.createBindGroupLayout(device, 1),
    entries: [],
  });

  // Aniso base scratch: .xyz = directional level-0 dims, .w = iso source mip (always 0).
  const anisoBaseArr = getTypeTypedArray(anisoBaseMeta.uniforms.dst.type); // Int32Array(4)
  // Aniso volume scratch: .xyz = destination level dims (re-uploaded per level, per-level buffers).
  const anisoVolArr = getTypeTypedArray(anisoVolMeta.uniforms.dst.type); // Int32Array(4)

  // Aniso directional-pyramid state (rebuilt by rebindGrid). anisoBaseGroup* drive the BASE pass
  // (one dispatch, iso mip 0 → 6 directional level-0 volumes); anisoVolGroup*[c] downsample every
  // direction from level c to c+1 (mirrors mipBuf/mipGroup0/mipGroup2 exactly, ×6 textures).
  let anisoTex: AnisoTextures | undefined;
  let anisoBaseGroup0: GPUBindGroup;
  let anisoBaseGroup2: GPUBindGroup;
  let anisoBaseX = 0;
  let anisoBaseY = 0;
  let anisoBaseZ = 0;
  let anisoMipCount = 1;
  let anisoVolBuf: GPUBuffer[] = [];
  let anisoVolGroup0: GPUBindGroup[] = [];
  let anisoVolGroup2: GPUBindGroup[] = [];

  // ===== Anisotropic directional pyramid (rebuilt alongside the iso pyramid). =====
  // Six directional volumes at half the iso mip-0 resolution, each with its own mip chain. The
  // BASE pass reads iso mip 0; the VOLUME pass downsamples each direction level c → c+1. These
  // are recreated on grid change (the old set is destroyed here before the new set is created).
  function rebindGrid(voxelRadiance: GPUTexture, grid: AnisoGridBox) {
    const { originX, originY, originZ, dimX, dimY, dimZ, cellSize } = grid;
    // Destroy the previous directional volumes (if any) before recreating at the new dims.
    if (anisoTex) {
      anisoTex.negX.destroy();
      anisoTex.posX.destroy();
      anisoTex.negY.destroy();
      anisoTex.posY.destroy();
      anisoTex.negZ.destroy();
      anisoTex.posZ.destroy();
    }
    anisoTex = createAnisoTextures(device, {
      originX,
      originY,
      originZ,
      dimX,
      dimY,
      dimZ,
      cellSize,
    });
    const ab = anisoBaseDims(dimX, dimY, dimZ);
    anisoBaseX = ab.x;
    anisoBaseY = ab.y;
    anisoBaseZ = ab.z;
    anisoMipCount = voxelMipLevelCount(ab.x, ab.y, ab.z);

    // Directional volumes in the fixed −X,+X,−Y,+Y,−Z,+Z order the shaders declare them.
    const anisoDirs = [
      anisoTex.negX,
      anisoTex.posX,
      anisoTex.negY,
      anisoTex.posY,
      anisoTex.negZ,
      anisoTex.posZ,
    ];
    const anisoBaseBindings = [
      anisoBaseMeta.uniforms.dstNegX.binding,
      anisoBaseMeta.uniforms.dstPosX.binding,
      anisoBaseMeta.uniforms.dstNegY.binding,
      anisoBaseMeta.uniforms.dstPosY.binding,
      anisoBaseMeta.uniforms.dstNegZ.binding,
      anisoBaseMeta.uniforms.dstPosZ.binding,
    ];
    const anisoVolSrcBindings = [
      anisoVolMeta.uniforms.srcNegX.binding,
      anisoVolMeta.uniforms.srcPosX.binding,
      anisoVolMeta.uniforms.srcNegY.binding,
      anisoVolMeta.uniforms.srcPosY.binding,
      anisoVolMeta.uniforms.srcNegZ.binding,
      anisoVolMeta.uniforms.srcPosZ.binding,
    ];
    const anisoVolDstBindings = [
      anisoVolMeta.uniforms.dstNegX.binding,
      anisoVolMeta.uniforms.dstPosX.binding,
      anisoVolMeta.uniforms.dstNegY.binding,
      anisoVolMeta.uniforms.dstPosY.binding,
      anisoVolMeta.uniforms.dstNegZ.binding,
      anisoVolMeta.uniforms.dstPosZ.binding,
    ];

    // BASE group0 = uDst uniform + iso mip 0 (single-mip sampled view); group2 = the 6 directional
    // level-0 storage views. One dispatch fills all 6 directions from the iso base.
    anisoBaseGroup0 = device.createBindGroup({
      layout: anisoBasePipeline.getBindGroupLayout(0),
      entries: [
        anisoBaseShader.uniforms.dst.getBindGroupEntry(device),
        {
          binding: anisoBaseMeta.uniforms.srcIso.binding,
          resource: voxelRadiance.createView({
            dimension: "3d",
            baseMipLevel: 0,
            mipLevelCount: 1,
          }),
        },
      ],
    });
    anisoBaseGroup2 = device.createBindGroup({
      layout: anisoBasePipeline.getBindGroupLayout(2),
      entries: anisoDirs.map((tex, i) => ({
        binding: anisoBaseBindings[i],
        resource: tex.createView({ dimension: "3d", baseMipLevel: 0, mipLevelCount: 1 }),
      })),
    });
    anisoBaseArr[0] = ab.x;
    anisoBaseArr[1] = ab.y;
    anisoBaseArr[2] = ab.z;
    anisoBaseArr[3] = 0; // iso source mip level (always 0 — see voxelAnisoBase.shader.ts)
    device.queue.writeBuffer(anisoBaseShader.uniforms.dst.getGPUBuffer(device), 0, anisoBaseArr);

    // VOLUME groups: one (group0, group2) pair per level c → c+1, each with its own uDst buffer
    // (per-level dims), 6 single-mip source views (level c) + 6 single-mip storage views (level c+1).
    for (const b of anisoVolBuf) b.destroy();
    anisoVolBuf = [];
    anisoVolGroup0 = [];
    anisoVolGroup2 = [];
    for (let c = 0; c < anisoMipCount - 1; c++) {
      const buf = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      anisoVolBuf.push(buf);
      anisoVolGroup0.push(
        device.createBindGroup({
          layout: anisoVolPipeline.getBindGroupLayout(0),
          entries: [
            { binding: anisoVolMeta.uniforms.dst.binding, resource: { buffer: buf } },
            ...anisoDirs.map((tex, i) => ({
              binding: anisoVolSrcBindings[i],
              resource: tex.createView({ dimension: "3d", baseMipLevel: c, mipLevelCount: 1 }),
            })),
          ],
        }),
      );
      anisoVolGroup2.push(
        device.createBindGroup({
          layout: anisoVolPipeline.getBindGroupLayout(2),
          entries: anisoDirs.map((tex, i) => ({
            binding: anisoVolDstBindings[i],
            resource: tex.createView({ dimension: "3d", baseMipLevel: c + 1, mipLevelCount: 1 }),
          })),
        }),
      );
      anisoVolArr[0] = Math.max(1, ab.x >> (c + 1));
      anisoVolArr[1] = Math.max(1, ab.y >> (c + 1));
      anisoVolArr[2] = Math.max(1, ab.z >> (c + 1));
      anisoVolArr[3] = 0;
      device.queue.writeBuffer(buf, 0, anisoVolArr);
    }
  }

  function base(encoder: GPUCommandEncoder) {
    runAnisoBase(encoder, {
      pipeline: anisoBasePipeline,
      emptyGroup1: anisoBaseEmptyGroup1,
      group0: anisoBaseGroup0,
      group2: anisoBaseGroup2,
      baseX: anisoBaseX,
      baseY: anisoBaseY,
      baseZ: anisoBaseZ,
      workgroup: ANISO_WG,
    });
  }

  function mips(encoder: GPUCommandEncoder) {
    runAnisoMips(encoder, {
      pipeline: anisoVolPipeline,
      emptyGroup1: anisoVolEmptyGroup1,
      group0: anisoVolGroup0,
      group2: anisoVolGroup2,
      baseX: anisoBaseX,
      baseY: anisoBaseY,
      baseZ: anisoBaseZ,
      count: anisoMipCount,
      workgroup: ANISO_WG,
    });
  }

  // The 6 directional volumes — read by the screen-probe GATHER (bound into its group0). The gather
  // groups are rebuilt (buildScreenProbeGroups) after rebindGrid recreates these, so callers read
  // the fresh set through this getter.
  function getTextures(): AnisoTextures {
    return anisoTex!;
  }

  return {
    base,
    mips,
    rebindGrid,
    getTextures,
  };
}
