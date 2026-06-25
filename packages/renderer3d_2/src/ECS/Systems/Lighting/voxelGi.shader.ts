import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";

// Stage 2.1a — brute-force voxel GI (the reference the cascades must converge to).
// Fullscreen pass, fully voxel-based (no G-buffer → no zoom-in fill cost):
//   1. Primary camera ray → DDA to the first solid voxel (its world pos + face normal +
//      albedo + emission).
//   2. At the hit, cast K cosine-weighted hemisphere rays around the face normal; each
//      DDA-traces to the first solid voxel and returns its emission (0 for occluders) or
//      the sky on a miss. Average = incoming diffuse radiance (single bounce).
//   3. lit = albedo*(ambient + GI*giStrength) + selfEmission.
// No temporal accumulation yet (Stage 2.1b) — static per-pixel jitter (no shimmer).

const sampled3d = (name: string) =>
  new VariableMeta(name, VariableKind.Texture, `texture_3d<f32>`, {
    viewDimension: "3d",
    textureSampleType: "float",
  });

export const shaderMeta = new ShaderMeta(
  {
    // .x = ambient, .y = numRays, .z = maxDist (world), .w = normalBias.
    params: new VariableMeta("uParams", VariableKind.Uniform, `vec4<f32>`),
    // .x = frameIndex (jitter seed), .y = skyIntensity (miss radiance), .z = giStrength.
    params2: new VariableMeta("uParams2", VariableKind.Uniform, `vec4<f32>`),
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
const PI: f32 = 3.14159265;
const BG = vec3<f32>(0.043, 0.051, 0.07);

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

fn unproject(ndc: vec3<f32>) -> vec3<f32> {
  let w = uInvViewProj * vec4<f32>(ndc, 1.0);
  return w.xyz / w.w;
}

// PCG hash → uniform float in [0,1).
fn pcg(v_in: u32) -> u32 {
  var v = v_in * 747796405u + 2891336453u;
  let s = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
  return (s >> 22u) ^ s;
}

// Orthonormal basis with column 2 = n (so basis * (x,y,z) = x*t + y*b + z*n).
fn build_basis(n: vec3<f32>) -> mat3x3<f32> {
  let a = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.9);
  let t = normalize(cross(a, n));
  let b = cross(n, t);
  return mat3x3<f32>(t, b, n);
}

// DDA (Amanatides–Woo) to the first solid voxel (occupancy a>0.5) within maxT world
// units. hit=false on a miss / exit / out-of-range.
struct Hit { hit: bool, coord: vec3<i32>, normal: vec3<f32> };

fn dda(ro: vec3<f32>, rd: vec3<f32>, maxT: f32) -> Hit {
  var res: Hit;
  res.hit = false;
  res.coord = vec3<i32>(0);
  res.normal = vec3<f32>(0.0, 0.0, 1.0);

  let cellSize = uGridOrigin.w;
  let gridMin = uGridOrigin.xyz;
  let dims = uGridDims.xyz;
  let gridMax = gridMin + vec3<f32>(dims) * cellSize;

  let inv = 1.0 / rd;
  let ta = (gridMin - ro) * inv;
  let tb = (gridMax - ro) * inv;
  let tsmall = min(ta, tb);
  let tbig = max(ta, tb);
  let tEnter = max(max(tsmall.x, tsmall.y), tsmall.z);
  let tExit = min(min(tbig.x, tbig.y), tbig.z);
  if (tExit < max(tEnter, 0.0)) { return res; }

  let tStart = max(tEnter, 0.0);
  if (tStart > maxT) { return res; }

  let pEnter = ro + rd * (tStart + cellSize * 1e-3);
  var coord = clamp(
    vec3<i32>(floor((pEnter - gridMin) / cellSize)),
    vec3<i32>(0),
    dims - vec3<i32>(1),
  );

  let step = vec3<i32>(sign(rd));
  let tDelta = abs(cellSize * inv);
  let nextBoundary = gridMin + (vec3<f32>(coord) + max(vec3<f32>(step), vec3<f32>(0.0))) * cellSize;
  var tMax = (nextBoundary - ro) * inv;

  var normal = vec3<f32>(0.0, 0.0, 1.0);
  if (tsmall.x >= tsmall.y && tsmall.x >= tsmall.z) {
    normal = vec3<f32>(-f32(step.x), 0.0, 0.0);
  } else if (tsmall.y >= tsmall.z) {
    normal = vec3<f32>(0.0, -f32(step.y), 0.0);
  } else {
    normal = vec3<f32>(0.0, 0.0, -f32(step.z));
  }

  let maxSteps = dims.x + dims.y + dims.z;
  var tCur = tStart;
  for (var i = 0; i < maxSteps; i = i + 1) {
    if (textureLoad(voxelAlbedo, coord, 0).a > 0.5) {
      res.hit = true;
      res.coord = coord;
      res.normal = normal;
      return res;
    }
    if (tMax.x <= tMax.y && tMax.x <= tMax.z) {
      tCur = tMax.x;
      coord.x = coord.x + step.x;
      tMax.x = tMax.x + tDelta.x;
      normal = vec3<f32>(-f32(step.x), 0.0, 0.0);
      if (coord.x < 0 || coord.x >= dims.x) { return res; }
    } else if (tMax.y <= tMax.z) {
      tCur = tMax.y;
      coord.y = coord.y + step.y;
      tMax.y = tMax.y + tDelta.y;
      normal = vec3<f32>(0.0, -f32(step.y), 0.0);
      if (coord.y < 0 || coord.y >= dims.y) { return res; }
    } else {
      tCur = tMax.z;
      coord.z = coord.z + step.z;
      tMax.z = tMax.z + tDelta.z;
      normal = vec3<f32>(0.0, 0.0, -f32(step.z));
      if (coord.z < 0 || coord.z >= dims.z) { return res; }
    }
    if (tCur > maxT) { return res; }
  }
  return res;
}

// Incoming radiance along a secondary ray: emission of the first solid voxel hit, or the
// sky on a miss within maxDist.
fn trace_radiance(origin: vec3<f32>, dir: vec3<f32>, maxDist: f32) -> vec3<f32> {
  let h = dda(origin, dir, maxDist);
  if (h.hit) {
    return textureLoad(voxelEmission, h.coord, 0).rgb;
  }
  return vec3<f32>(uParams2.y);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  // WebGPU clip Y is UP; texCoord V grows DOWN. Reverse-Z: NEAR->1, FAR->0.
  let ndcXY = vec2<f32>(input.texCoord.x * 2.0 - 1.0, (1.0 - input.texCoord.y) * 2.0 - 1.0);
  let nearW = unproject(vec3<f32>(ndcXY, 1.0));
  let farW = unproject(vec3<f32>(ndcXY, 0.0));
  let ro = nearW;
  let rd = normalize(farW - nearW);

  let prim = dda(ro, rd, 1e9);
  if (!prim.hit) {
    return vec4f(BG, 1.0);
  }

  let cellSize = uGridOrigin.w;
  let hitWorld = uGridOrigin.xyz + (vec3<f32>(prim.coord) + vec3<f32>(0.5)) * cellSize;
  let N = prim.normal;
  let albedo = textureLoad(voxelAlbedo, prim.coord, 0).rgb;
  let selfEmission = textureLoad(voxelEmission, prim.coord, 0).rgb;

  // Lift the secondary origin off the surface to avoid hitting coplanar solid voxels.
  let origin = hitWorld + N * (cellSize * 1.5 + uParams.w);
  let basis = build_basis(N);

  let numRays = max(1u, u32(uParams.y));
  let maxDist = uParams.z;
  let frame = u32(uParams2.x);
  let px = vec2<u32>(vec2<i32>(floor(input.position.xy)));
  var seed = (px.x * 1973u + px.y * 9277u + frame * 26699u) | 1u;

  var gi = vec3<f32>(0.0);
  for (var i = 0u; i < numRays; i = i + 1u) {
    seed = pcg(seed);
    let u1 = f32(seed) / 4294967296.0;
    seed = pcg(seed);
    let u2 = f32(seed) / 4294967296.0;
    // Cosine-weighted hemisphere sample (pdf = cos/PI cancels the integrand → average).
    let r = sqrt(u1);
    let phi = 2.0 * PI * u2;
    let local = vec3<f32>(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));
    let dir = normalize(basis * local);
    gi = gi + trace_radiance(origin, dir, maxDist);
  }
  gi = gi / f32(numRays);

  let lit = albedo * (uParams.x + gi * uParams2.z) + selfEmission;
  return vec4f(lit, 1.0);
}
`,
);
