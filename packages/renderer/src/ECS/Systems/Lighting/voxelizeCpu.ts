import type { SceneInstances } from "../SDFSystem/createDrawShapeSystem.ts";

// CPU-side work for the voxelize pass: per-instance conservative voxel-AABB computation + the
// scatter work-list prefix sum. Pure (reads the scene-instance CPU mirrors + grid params, writes the
// preallocated aabbMin/aabbDim scratch), so it lives here instead of inside the pass dispatch.

// Half the shape's Z extent from its per-kind depth slot in Shape.values (stride 8).
// Mirrors footprint_half_z in sceneSDF.wgsl; sphere (6) is unhalved (radius == half-extent).
export function footprintHalfZ(kind: number, values: Float32Array, k: number): number {
  switch (kind) {
    case 0:
      return values[k * 8 + 1] * 0.5;
    case 1:
      return values[k * 8 + 2] * 0.5;
    case 3:
    case 4:
      return values[k * 8 + 3] * 0.5;
    case 5:
      return values[k * 8 + 6] * 0.5;
    default:
      return values[k * 8 + 0];
  }
}

export type VoxelGridBox = {
  originX: number;
  originY: number;
  originZ: number;
  cellSize: number;
  dimX: number;
  dimY: number;
  dimZ: number;
};

// Build per-instance voxel AABBs + the prefix-sum work list for the scatter pass. For each instance:
// compute a CONSERVATIVE rotated-box world AABB, convert it to a voxel box clamped to the grid, and
// accumulate the prefix sum of voxel counts. The scatter shader walks this flat list via a binary
// search on `start`. aabbMin.w = prefix start; aabbDim.w = voxel count. Returns scatterTotal (the
// grand total of scattered voxels = the last prefix).
export function buildVoxelAABBs(
  scene: SceneInstances,
  box: VoxelGridBox,
  aabbMinArr: Int32Array,
  aabbDimArr: Int32Array,
): number {
  const { originX, originY, originZ, cellSize, dimX, dimY, dimZ } = box;
  const n = scene.instanceCount;
  const tr = scene.cpuTransform;
  const kindArr = scene.cpuKind;
  const valArr = scene.cpuValues;
  const roundArr = scene.cpuRoundness;
  let prefix = 0;
  for (let k = 0; k < n; k++) {
    // Translation is column-major mat4 elements 12,13,14 (per-instance 16-float stride).
    const cx = tr[k * 16 + 12];
    const cy = tr[k * 16 + 13];
    const tz = tr[k * 16 + 14];
    const kind = kindArr[k];
    const round = roundArr[k];

    // Conservative bounding-circle radius in XY, computed from the SAME geometry the footprint uses
    // so the AABB never clips the shape. The depth slot must be EXCLUDED from the XY bound, so each
    // kind reads only its footprint (XY) slots — mirroring footprint_half_xy in sceneSDF.wgsl.
    const v0 = valArr[k * 8 + 0];
    const v1 = valArr[k * 8 + 1];
    const v2 = valArr[k * 8 + 2];
    let rxyShape: number;
    if (kind === 3) {
      // Parallelogram: skew widens the X half-extent (halfX = width/2 + |skew|); the worst
      // corner is at hypot(halfX, height/2). values = (width, height, skew, depth).
      rxyShape = Math.hypot(v0 / 2 + Math.abs(v2), v1 / 2);
    } else if (kind === 5) {
      // Triangle: the first 6 slots are signed vertex coords (ax,ay,bx,by,cx,cy). The
      // conservative radius is the farthest vertex distance from the local origin.
      rxyShape = Math.max(
        Math.hypot(valArr[k * 8 + 0], valArr[k * 8 + 1]),
        Math.hypot(valArr[k * 8 + 2], valArr[k * 8 + 3]),
        Math.hypot(valArr[k * 8 + 4], valArr[k * 8 + 5]),
      );
    } else if (kind === 0 || kind === 6) {
      // Circle/cylinder + sphere: values[0] = radius (the full XY half-extent).
      rxyShape = v0;
    } else if (kind === 4) {
      // Trapezoid: values = [topWidth, bottomWidth, ySize, depth]. Bound = wider end / 2 in X,
      // ySize / 2 in Y → the worst corner is at hypot of those.
      rxyShape = Math.hypot(Math.max(v0, v1) / 2, v2 / 2);
    } else {
      // Rectangle/box: values = [width, height, depth]. Corner at hypot(width/2, height/2).
      rxyShape = Math.hypot(v0 / 2, v1 / 2);
    }
    // Local conservative half-extents: a single bounding-circle radius for X and Y (yaw-invariant)
    // plus the per-kind Z half. Under full rotation each axis grows by the rotated box bound
    // half_world = abs(R) * half_local (R = the instance's 3x3 rotation, column-major in tr:
    // element (row r, col c) = tr[k*16 + c*4 + r]).
    const hLocalXY = rxyShape;
    const hLocalZ = footprintHalfZ(kind, valArr, k);
    const m0 = Math.abs(tr[k * 16 + 0]);
    const m1 = Math.abs(tr[k * 16 + 1]);
    const m2 = Math.abs(tr[k * 16 + 2]);
    const m4 = Math.abs(tr[k * 16 + 4]);
    const m5 = Math.abs(tr[k * 16 + 5]);
    const m6 = Math.abs(tr[k * 16 + 6]);
    const m8 = Math.abs(tr[k * 16 + 8]);
    const m9 = Math.abs(tr[k * 16 + 9]);
    const m10 = Math.abs(tr[k * 16 + 10]);
    // Raw matrix columns already carry the uniform scale s (length of each column), so the
    // m*hLocal products are already abs(R)*(hLocal*s). The `round` term is in unscaled-local
    // units and must scale by s too; `cellSize` is a grid constant and stays unscaled.
    const s = Math.hypot(tr[k * 16 + 0], tr[k * 16 + 1], tr[k * 16 + 2]);
    const halfWX = m0 * hLocalXY + m4 * hLocalXY + m8 * hLocalZ + round * s + cellSize;
    const halfWY = m1 * hLocalXY + m5 * hLocalXY + m9 * hLocalZ + round * s + cellSize;
    const halfWZ = m2 * hLocalXY + m6 * hLocalXY + m10 * hLocalZ + round * s + cellSize;

    const minX = cx - halfWX;
    const maxX = cx + halfWX;
    const minY = cy - halfWY;
    const maxY = cy + halfWY;
    const minZ = tz - halfWZ;
    const maxZ = tz + halfWZ;

    // World AABB -> voxel index box, clamped to [0, dim] (floor min, ceil max), then size.
    const vx0 = Math.min(Math.max(Math.floor((minX - originX) / cellSize), 0), dimX);
    const vx1 = Math.min(Math.max(Math.ceil((maxX - originX) / cellSize), 0), dimX);
    const vy0 = Math.min(Math.max(Math.floor((minY - originY) / cellSize), 0), dimY);
    const vy1 = Math.min(Math.max(Math.ceil((maxY - originY) / cellSize), 0), dimY);
    const vz0 = Math.min(Math.max(Math.floor((minZ - originZ) / cellSize), 0), dimZ);
    const vz1 = Math.min(Math.max(Math.ceil((maxZ - originZ) / cellSize), 0), dimZ);
    const nx = Math.max(0, vx1 - vx0);
    const ny = Math.max(0, vy1 - vy0);
    const nz = Math.max(0, vz1 - vz0);
    const count = nx * ny * nz;

    // Every index gets a `start` (empty ranges share their successor's start and are skipped
    // by the binary search). aabbMin.w = prefix start; aabbDim.w = voxel count.
    aabbMinArr[k * 4 + 0] = vx0;
    aabbMinArr[k * 4 + 1] = vy0;
    aabbMinArr[k * 4 + 2] = vz0;
    aabbMinArr[k * 4 + 3] = prefix;
    aabbDimArr[k * 4 + 0] = nx;
    aabbDimArr[k * 4 + 1] = ny;
    aabbDimArr[k * 4 + 2] = nz;
    aabbDimArr[k * 4 + 3] = count;
    prefix += count;
  }
  return prefix;
}
