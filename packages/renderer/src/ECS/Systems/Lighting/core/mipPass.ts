import { gpuSpan } from "../../../../gpuTimer.ts";

// Mip-pyramid + anisotropic-pyramid downsample passes. Each is thin GPU-dispatch glue that issues
// one compute pass per pyramid level. Extracted from createVoxelSystem as standalone functions that
// take a scoped deps bundle (exactly the handles + dims they need) — the caller owns the state and
// builds the bundle at the call site; no shared mutable context.

export type MipPassDeps = {
  pipeline: GPUComputePipeline;
  emptyGroup1: GPUBindGroup;
  // group0[L]/group2[L] downsample iso mip L → mip L+1.
  group0: GPUBindGroup[];
  group2: GPUBindGroup[];
  count: number; // mip level count (count-1 downsample steps)
  dimX: number;
  dimY: number;
  dimZ: number;
  workgroup: number;
  // GPU-timing label — every per-level pass tags itself with it; the profiler sums same-label
  // spans into one row (see gpuTimer.ts).
  label: string;
};

// Build the voxelRadiance mip pyramid: one compute pass PER level (encoder-barriered, so level L+1
// sees level L's writes). MUST run AFTER voxelize (level 0 reads mip 0 that voxelize wrote).
export function runMips(encoder: GPUCommandEncoder, d: MipPassDeps): void {
  for (let L = 0; L < d.count - 1; L++) {
    const dx = Math.max(1, d.dimX >> (L + 1));
    const dy = Math.max(1, d.dimY >> (L + 1));
    const dz = Math.max(1, d.dimZ >> (L + 1));
    const pass = encoder.beginComputePass({ timestampWrites: gpuSpan(d.label) });
    pass.setPipeline(d.pipeline);
    pass.setBindGroup(0, d.group0[L]);
    pass.setBindGroup(1, d.emptyGroup1);
    pass.setBindGroup(2, d.group2[L]);
    pass.dispatchWorkgroups(
      Math.ceil(dx / d.workgroup),
      Math.ceil(dy / d.workgroup),
      Math.ceil(dz / d.workgroup),
    );
    pass.end();
  }
}

export type AnisoBasePassDeps = {
  pipeline: GPUComputePipeline;
  emptyGroup1: GPUBindGroup;
  group0: GPUBindGroup;
  group2: GPUBindGroup;
  baseX: number;
  baseY: number;
  baseZ: number;
  workgroup: number;
  label: string; // GPU-timing label (see MipPassDeps.label)
};

// Build the 6 directional level-0 volumes from iso voxelRadiance mip 0 (one compute pass). MUST run
// AFTER voxelize (reads iso mip 0). Followed by runAnisoMips for the coarser levels.
export function runAnisoBase(encoder: GPUCommandEncoder, d: AnisoBasePassDeps): void {
  const pass = encoder.beginComputePass({ timestampWrites: gpuSpan(d.label) });
  pass.setPipeline(d.pipeline);
  pass.setBindGroup(0, d.group0);
  pass.setBindGroup(1, d.emptyGroup1);
  pass.setBindGroup(2, d.group2);
  pass.dispatchWorkgroups(
    Math.ceil(d.baseX / d.workgroup),
    Math.ceil(d.baseY / d.workgroup),
    Math.ceil(d.baseZ / d.workgroup),
  );
  pass.end();
}

export type AnisoMipsPassDeps = {
  pipeline: GPUComputePipeline;
  emptyGroup1: GPUBindGroup;
  group0: GPUBindGroup[];
  group2: GPUBindGroup[];
  baseX: number;
  baseY: number;
  baseZ: number;
  count: number; // aniso mip level count (count-1 downsample steps)
  workgroup: number;
  label: string; // GPU-timing label (see MipPassDeps.label)
};

// Downsample every directional volume level c → c+1 (one compute pass per level, encoder-barriered
// so c+1 sees c's writes). MUST run AFTER runAnisoBase (level 0 must exist).
export function runAnisoMips(encoder: GPUCommandEncoder, d: AnisoMipsPassDeps): void {
  for (let c = 0; c < d.count - 1; c++) {
    const dx = Math.max(1, d.baseX >> (c + 1));
    const dy = Math.max(1, d.baseY >> (c + 1));
    const dz = Math.max(1, d.baseZ >> (c + 1));
    const pass = encoder.beginComputePass({ timestampWrites: gpuSpan(d.label) });
    pass.setPipeline(d.pipeline);
    pass.setBindGroup(0, d.group0[c]);
    pass.setBindGroup(1, d.emptyGroup1);
    pass.setBindGroup(2, d.group2[c]);
    pass.dispatchWorkgroups(
      Math.ceil(dx / d.workgroup),
      Math.ceil(dy / d.workgroup),
      Math.ceil(dz / d.workgroup),
    );
    pass.end();
  }
}
