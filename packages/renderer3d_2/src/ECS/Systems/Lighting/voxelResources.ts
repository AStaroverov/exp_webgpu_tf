// Voxel-grid GPU textures + grid configuration. The grid is an axis-aligned box in
// WORLD space (Z-up): origin = its min corner, dims = voxel counts per axis, cellSize =
// world units per voxel (cubic). Voxel (i,j,k) center = origin + (vec3(i,j,k)+0.5)*cellSize.
//
// Three 3D storage textures, both STORAGE_BINDING (written by the voxelize compute pass via
// textureStore) + TEXTURE_BINDING (read by later passes as a sampled texture_3d<f32>):
//   voxelAlbedo   rgba8unorm  — rgb = nearest-instance albedo, a = occupancy (1=solid).
//   voxelEmission rgba16float — rgb = nearest-instance emission (for the Phase-2 GI).
//   voxelRadiance rgba16float — rgb = direct-lit radiance (sun N·L·vis) + emission, a = occupancy.
//
// voxelRadiance carries a FULL MIP PYRAMID (voxelMipLevelCount levels): mip 0 is written by
// the voxelize pass, the coarser levels by the voxelMip compute pass (isotropic, opacity-
// weighted downsample). The pyramid is what the VCT cone-tracing path samples by LOD.
// voxelAlbedo / voxelEmission stay single-mip.

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

// Number of mip levels for the voxelRadiance pyramid: full chain down to the 1-voxel mip
// along the LARGEST axis (1 + floor(log2(max dim))).
export function voxelMipLevelCount(dimX: number, dimY: number, dimZ: number): number {
  return 1 + Math.floor(Math.log2(Math.max(dimX, dimY, dimZ)));
}

// ===== Anisotropic VCT (Layer 6) — 6 directional radiance volumes (±X,±Y,±Z). =====
// To make occlusion DIRECTION-correct (and stop bright emitters leaking through occluders at
// coarse LOD), the radiance pyramid is also stored anisotropically: six 3D volumes, each built
// with a FRONT-TO-BACK pre-integration along its axis. They start at HALF the base resolution
// (= iso mip1 dims) with their OWN mip pyramids — the standard memory-saver. The cone trace
// picks the 3 volumes facing back toward the cone direction and blends them (Crassin).

// Directional-volume level-0 dims = half the base grid (ceil), i.e. the iso mip1 footprint.
export function voxelDirDims(
  dimX: number,
  dimY: number,
  dimZ: number,
): { x: number; y: number; z: number } {
  return {
    x: Math.max(1, Math.ceil(dimX / 2)),
    y: Math.max(1, Math.ceil(dimY / 2)),
    z: Math.max(1, Math.ceil(dimZ / 2)),
  };
}

// Mip level count of a directional pyramid (computed from its OWN half-res dims — DIFFERENT
// from the iso voxelMipLevelCount(base dims); the shorter Z bottoms out earlier).
export function voxelDirLevelCount(dimX: number, dimY: number, dimZ: number): number {
  const d = voxelDirDims(dimX, dimY, dimZ);
  return 1 + Math.floor(Math.log2(Math.max(d.x, d.y, d.z)));
}

export type VoxelTextures = {
  voxelAlbedo: GPUTexture;
  voxelEmission: GPUTexture;
  voxelRadiance: GPUTexture;
  // 6 directional radiance volumes (front-to-back pre-integrated), half-res with own mips.
  // Index/order matches the shader's dir convention: 0=-X,1=+X,2=-Y,3=+Y,4=-Z,5=+Z.
  voxelDirNegX: GPUTexture;
  voxelDirPosX: GPUTexture;
  voxelDirNegY: GPUTexture;
  voxelDirPosY: GPUTexture;
  voxelDirNegZ: GPUTexture;
  voxelDirPosZ: GPUTexture;
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
  const voxelRadiance = device.createTexture({
    size,
    dimension: "3d",
    format: "rgba16float",
    mipLevelCount: voxelMipLevelCount(grid.dimX, grid.dimY, grid.dimZ),
    usage,
  });

  // 6 directional volumes at half base resolution, each with its own mip pyramid.
  const dir = voxelDirDims(grid.dimX, grid.dimY, grid.dimZ);
  const dirSize: [number, number, number] = [dir.x, dir.y, dir.z];
  const dirMips = voxelDirLevelCount(grid.dimX, grid.dimY, grid.dimZ);
  const createDir = () =>
    device.createTexture({
      size: dirSize,
      dimension: "3d",
      format: "rgba16float",
      mipLevelCount: dirMips,
      usage,
    });
  const voxelDirNegX = createDir();
  const voxelDirPosX = createDir();
  const voxelDirNegY = createDir();
  const voxelDirPosY = createDir();
  const voxelDirNegZ = createDir();
  const voxelDirPosZ = createDir();

  return {
    voxelAlbedo,
    voxelEmission,
    voxelRadiance,
    voxelDirNegX,
    voxelDirPosX,
    voxelDirNegY,
    voxelDirPosY,
    voxelDirNegZ,
    voxelDirPosZ,
  };
}
