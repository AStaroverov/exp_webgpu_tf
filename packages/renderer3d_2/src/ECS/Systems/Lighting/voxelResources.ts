// Voxel-grid GPU textures + grid configuration. The grid is an axis-aligned box in
// WORLD space (Z-up): origin = its min corner, dims = voxel counts per axis, cellSize =
// world units per voxel (cubic). Voxel (i,j,k) center = origin + (vec3(i,j,k)+0.5)*cellSize.
//
// One 3D storage texture, both STORAGE_BINDING (written by the voxelize compute pass via
// textureStore) + TEXTURE_BINDING (read by later passes as a sampled texture_3d<f32>):
//   voxelRadiance rgba16float — rgb = direct-lit radiance (sun N·L·vis) + emission, a = occupancy.
//
// voxelRadiance carries a FULL MIP PYRAMID (voxelMipLevelCount levels): mip 0 is written by
// the voxelize pass, the coarser levels by the voxelMip compute pass (isotropic, opacity-
// weighted downsample). The pyramid is what the VCT cone-tracing path samples by LOD.
//
// (The former voxelAlbedo / voxelEmission 3D volumes were removed: their only reader was the
// unimported voxelTrace.wgsl.ts; the composite reads the 2D G-buffer emission, not a volume.)

export type VoxelGridConfig = {
  // World-space min corner of the grid box.
  originX: number;
  originY: number;
  originZ: number;
  // Voxel counts per axis.
  dimX: number;
  dimY: number;
  dimZ: number;
  // World units per voxel (cubic).
  cellSize: number;
};

// World box x,y in [-32, 32], z in [-2, 14] (extent 64×64×16). At cellSize 0.25 that is
// 256×256×64 = 4 194 304 voxels → voxelRadiance (rgba16float + mip pyramid) ≈ 38 MB. Fine enough
// to resolve small objects in GI; raise cellSize via the GUI for coarser/cheaper, lower for finer.
export const DEFAULT_VOXEL_GRID: VoxelGridConfig = {
  originX: -32,
  originY: -32,
  originZ: -2,
  dimX: 256,
  dimY: 256,
  dimZ: 64,
  cellSize: 0.25,
};

// Number of mip levels for the voxelRadiance pyramid: full chain down to the 1-voxel mip
// along the LARGEST axis (1 + floor(log2(max dim))).
export function voxelMipLevelCount(dimX: number, dimY: number, dimZ: number): number {
  return 1 + Math.floor(Math.log2(Math.max(dimX, dimY, dimZ)));
}

export type VoxelTextures = {
  voxelRadiance: GPUTexture;
};

export function createVoxelTextures(
  device: GPUDevice,
  grid: VoxelGridConfig = DEFAULT_VOXEL_GRID,
): VoxelTextures {
  const size: [number, number, number] = [grid.dimX, grid.dimY, grid.dimZ];
  const usage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;

  const voxelRadiance = device.createTexture({
    size,
    dimension: "3d",
    format: "rgba16float",
    mipLevelCount: voxelMipLevelCount(grid.dimX, grid.dimY, grid.dimZ),
    usage,
  });

  return { voxelRadiance };
}
