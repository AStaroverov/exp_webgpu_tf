// Voxel-grid GPU textures + grid configuration. The grid is an axis-aligned box in
// WORLD space (Z-up): origin = its min corner, dims = voxel counts per axis, cellSize =
// world units per voxel (cubic). Voxel (i,j,k) center = origin + (vec3(i,j,k)+0.5)*cellSize.
//
// Two 3D storage textures, both STORAGE_BINDING (written by the voxelize compute pass via
// textureStore) + TEXTURE_BINDING (read by later passes as a sampled texture_3d<f32>):
//   voxelAlbedo   rgba8unorm  — rgb = nearest-instance albedo, a = occupancy (1=solid).
//   voxelEmission rgba16float — rgb = nearest-instance emission (for the Phase-2 GI).

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

// Covers the showcase scene: x,y in [-32, 32], z in [-2, 14]. 128*128*32 = 524 288 voxels
// (albedo 2 MB, emission 4 MB).
export const DEFAULT_VOXEL_GRID: VoxelGridConfig = {
  originX: -32,
  originY: -32,
  originZ: -2,
  dimX: 128,
  dimY: 128,
  dimZ: 32,
  cellSize: 0.5,
};

export type VoxelTextures = {
  voxelAlbedo: GPUTexture;
  voxelEmission: GPUTexture;
};

export function createVoxelTextures(
  device: GPUDevice,
  grid: VoxelGridConfig = DEFAULT_VOXEL_GRID,
): VoxelTextures {
  const size: [number, number, number] = [grid.dimX, grid.dimY, grid.dimZ];
  const usage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;

  const voxelAlbedo = device.createTexture({
    size,
    dimension: "3d",
    format: "rgba8unorm",
    usage,
  });
  const voxelEmission = device.createTexture({
    size,
    dimension: "3d",
    format: "rgba16float",
    usage,
  });

  return { voxelAlbedo, voxelEmission };
}
