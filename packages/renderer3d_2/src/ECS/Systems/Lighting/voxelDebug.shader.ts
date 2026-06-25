import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";

// Debug visualization of the voxel grid: a fullscreen pass that reconstructs the world-
// space camera ray per pixel (from invViewProj), DDA-marches (Amanatides–Woo) through the
// voxel grid, and shades the first solid voxel (occupancy a>0) with a simple Lambert term
// on the crossed face normal + its emission. A miss returns the background color.
//
// The voxel textures are read as sampled texture_3d<f32> via textureLoad (exact integer
// fetch, no filtering) — the SAME GPUTextures the voxelize compute pass wrote.

const sampled3d = (name: string) =>
  new VariableMeta(name, VariableKind.Texture, `texture_3d<f32>`, {
    viewDimension: "3d",
    textureSampleType: "float",
  });

export const shaderMeta = new ShaderMeta(
  {
    // .x = ambient floor. (y/z/w spare.)
    params: new VariableMeta("uParams", VariableKind.Uniform, `vec4<f32>`),
    // inverse(viewProjMatrix) (reverse-Z), column-major, for world-ray reconstruction.
    invViewProj: new VariableMeta("uInvViewProj", VariableKind.Uniform, `mat4x4<f32>`),
    // .xyz = world min corner, .w = cellSize.
    gridOrigin: new VariableMeta("uGridOrigin", VariableKind.Uniform, `vec4<f32>`),
    // .xyz = voxel counts per axis.
    gridDims: new VariableMeta("uGridDims", VariableKind.Uniform, `vec4<i32>`),
    voxelAlbedo: sampled3d("voxelAlbedo"),
    voxelEmission: sampled3d("voxelEmission"),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
const BG = vec3<f32>(0.043, 0.051, 0.07);
const LIGHT_DIR = vec3<f32>(0.4, 0.35, 1.0);

const POSITION = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
  vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);
const TEX_COORDS = array<vec2f, 6>(
  vec2f(0.0, 1.0), vec2f(1.0, 1.0), vec2f(1.0, 0.0),
  vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(0.0, 0.0)
);

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) texCoord: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var out: VertexOutput;
  out.position = vec4f(POSITION[vertexIndex], 0.0, 1.0);
  out.texCoord = TEX_COORDS[vertexIndex];
  return out;
}

// Unproject an NDC point (z reverse-Z) to world space.
fn unproject(ndc: vec3<f32>) -> vec3<f32> {
  let w = uInvViewProj * vec4<f32>(ndc, 1.0);
  return w.xyz / w.w;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  // WebGPU clip Y is UP; texCoord V grows DOWN, so NDC.y = (1 - v)*2 - 1.
  let ndcXY = vec2<f32>(input.texCoord.x * 2.0 - 1.0, (1.0 - input.texCoord.y) * 2.0 - 1.0);
  // Reverse-Z: NEAR -> 1, FAR -> 0. Two depths give a world ray (handles ortho too).
  let nearW = unproject(vec3<f32>(ndcXY, 1.0));
  let farW = unproject(vec3<f32>(ndcXY, 0.0));
  let ro = nearW;
  let rd = normalize(farW - nearW);

  let cellSize = uGridOrigin.w;
  let gridMin = uGridOrigin.xyz;
  let dims = uGridDims.xyz;
  let gridMax = gridMin + vec3<f32>(dims) * cellSize;

  // Ray vs grid AABB (slab test). inv may be +-inf when rd has a zero component; the
  // min/max ordering still yields the correct overlap interval.
  let inv = 1.0 / rd;
  let t0 = (gridMin - ro) * inv;
  let t1 = (gridMax - ro) * inv;
  let tsmall = min(t0, t1);
  let tbig = max(t0, t1);
  let tEnter = max(max(tsmall.x, tsmall.y), tsmall.z);
  let tExit = min(min(tbig.x, tbig.y), tbig.z);
  if (tExit < max(tEnter, 0.0)) {
    return vec4f(BG, 1.0);
  }

  let tStart = max(tEnter, 0.0);
  // Nudge inside to avoid landing exactly on a boundary.
  let pEnter = ro + rd * (tStart + cellSize * 1e-3);
  var coord = vec3<i32>(floor((pEnter - gridMin) / cellSize));
  coord = clamp(coord, vec3<i32>(0), dims - vec3<i32>(1));

  let step = vec3<i32>(sign(rd));
  let tDelta = abs(cellSize * inv);
  // Distance along the ray to the first voxel boundary on each axis.
  let nextBoundary = gridMin + (vec3<f32>(coord) + max(vec3<f32>(step), vec3<f32>(0.0))) * cellSize;
  var tMax = (nextBoundary - ro) * inv;

  // Entry face normal: the axis that produced tEnter, pointing back along the ray.
  var normal = vec3<f32>(0.0, 0.0, 1.0);
  if (tsmall.x >= tsmall.y && tsmall.x >= tsmall.z) {
    normal = vec3<f32>(-f32(step.x), 0.0, 0.0);
  } else if (tsmall.y >= tsmall.z) {
    normal = vec3<f32>(0.0, -f32(step.y), 0.0);
  } else {
    normal = vec3<f32>(0.0, 0.0, -f32(step.z));
  }

  let maxSteps = dims.x + dims.y + dims.z;
  for (var i = 0; i < maxSteps; i = i + 1) {
    let a = textureLoad(voxelAlbedo, coord, 0);
    if (a.a > 0.5) {
      let N = normal;
      let diff = max(0.0, dot(N, normalize(LIGHT_DIR)));
      let emission = textureLoad(voxelEmission, coord, 0).rgb;
      let lit = a.rgb * (uParams.x + diff) + emission;
      return vec4f(lit, 1.0);
    }

    // Advance to the next voxel along the smallest tMax axis.
    if (tMax.x <= tMax.y && tMax.x <= tMax.z) {
      coord.x = coord.x + step.x;
      tMax.x = tMax.x + tDelta.x;
      normal = vec3<f32>(-f32(step.x), 0.0, 0.0);
      if (coord.x < 0 || coord.x >= dims.x) { break; }
    } else if (tMax.y <= tMax.z) {
      coord.y = coord.y + step.y;
      tMax.y = tMax.y + tDelta.y;
      normal = vec3<f32>(0.0, -f32(step.y), 0.0);
      if (coord.y < 0 || coord.y >= dims.y) { break; }
    } else {
      coord.z = coord.z + step.z;
      tMax.z = tMax.z + tDelta.z;
      normal = vec3<f32>(0.0, 0.0, -f32(step.z));
      if (coord.z < 0 || coord.z >= dims.z) { break; }
    }
  }

  return vec4f(BG, 1.0);
}
`,
);
