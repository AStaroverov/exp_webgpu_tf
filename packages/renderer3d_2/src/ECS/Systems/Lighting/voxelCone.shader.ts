import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";

// VCT Layer 3 — the full DIFFUSE HEMISPHERE cone gather. A fullscreen pass over the G-buffer:
//   1. Reconstruct the per-pixel world position P (from the reverse-Z depth + invViewProj)
//      and world normal N (from the normal G-buffer; n.rgb*2-1, n.a<0.5 = no surface).
//   2. Trace 6 cones over the hemisphere around N through the voxelRadiance mip pyramid:
//      1 cone along the normal + a ring of 5 tilted 60° off the normal (cosine-weighted).
//      Each cone's diameter grows with distance (diameter = 2*aperture*dist); each step
//      samples the pyramid at LOD = log2(diameter / voxelSize) — a wider cone reads a
//      coarser mip, so one filtered fetch integrates the whole cone cross-section.
//      Front-to-back "over" compositing accumulates radiance + opacity per cone. The cone
//      results are combined with cosine weights {π/4 (normal), 3π/20 ×5 (ring)} summing to
//      π, then normalized by π → the cosine-weighted AVERAGE incoming radiance.
//   3. Write the gathered radiance (scaled by giStrength) to a full-res HDR texture.
//
// This is what produces real color bleeding. The earlier single-cone-along-the-normal form
// was the Layer-2 INTERMEDIATE (a bent-normal / AO preview); the hemisphere gather here is
// directly comparable to the brute-force gi reference, which is also a cosine-weighted
// hemisphere average.
//
// Mirrors voxelDebug.shader.ts for the fullscreen setup + `unproject`, and voxelGi's
// reverse-Z NDC reconstruction. textureSampleLevel (explicit LOD) is used in the trace loop
// — legal in non-uniform control flow (unlike textureSample).

export const shaderMeta = new ShaderMeta(
  {
    // .x = normalBias, .y = maxDist (world), .z = aperture = tan(halfAngle), .w = giStrength.
    params: new VariableMeta("uParams", VariableKind.Uniform, `vec4<f32>`),
    // .x = screen width (px), .y = screen height (px). (z/w spare.)
    params2: new VariableMeta("uParams2", VariableKind.Uniform, `vec4<f32>`),
    // inverse(viewProjMatrix) (reverse-Z), column-major, for world-position reconstruction.
    invViewProj: new VariableMeta("uInvViewProj", VariableKind.Uniform, `mat4x4<f32>`),
    // .xyz = world min corner, .w = cellSize.
    gridOrigin: new VariableMeta("uGridOrigin", VariableKind.Uniform, `vec4<f32>`),
    // .xyz = voxel counts per axis.
    gridDims: new VariableMeta("uGridDims", VariableKind.Uniform, `vec4<i32>`),
    // G-buffer reverse-Z depth (texture_depth_2d) + world normal (rgba16float, packed *0.5+0.5).
    depthTex: new VariableMeta("depthTex", VariableKind.Texture, `texture_depth_2d`, {
      textureSampleType: "depth",
    }),
    normalTex: new VariableMeta("normalTex", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "float",
    }),
    // The voxelRadiance mip pyramid (ALL mips) — the cone reads it at the per-step LOD.
    voxelRadiance: new VariableMeta("voxelRadiance", VariableKind.Texture, `texture_3d<f32>`, {
      viewDimension: "3d",
      textureSampleType: "float",
    }),
    // Filtering sampler for textureSampleLevel over the pyramid.
    voxelSampler: new VariableMeta("voxelSampler", VariableKind.Sampler, `sampler`),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
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

const PI: f32 = 3.14159265;

// Unproject an NDC point (z reverse-Z) to world space.
fn unproject(ndc: vec3<f32>) -> vec3<f32> {
  let w = uInvViewProj * vec4<f32>(ndc, 1.0);
  return w.xyz / w.w;
}

// Orthonormal basis with column 2 = n (so basis * (x,y,z) = x*t + y*b + z*n).
fn build_basis(n: vec3<f32>) -> mat3x3<f32> {
  let a = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.9);
  let t = normalize(cross(a, n));
  let b = cross(n, t);
  return mat3x3<f32>(t, b, n);
}

// One cone marched along dir from origin: diameter grows with distance, each step samples
// voxelRadiance at LOD=log2(diameter/voxelSize) and composites front-to-back ("over").
fn trace_cone(origin: vec3<f32>, dir: vec3<f32>, aperture: f32) -> vec4<f32> {
  var col = vec3<f32>(0.0);
  var alpha = 0.0;
  let voxelSize = uGridOrigin.w;
  let gridMin = uGridOrigin.xyz;
  let extent = vec3<f32>(uGridDims.xyz) * voxelSize;
  var dist = voxelSize;
  let maxDist = uParams.y;
  for (var i = 0; i < 64; i = i + 1) {
    if (alpha >= 1.0 || dist > maxDist) { break; }
    let diameter = max(voxelSize, 2.0 * aperture * dist);
    let lod = log2(diameter / voxelSize);
    let wp = origin + dir * dist;
    let uvw = (wp - gridMin) / extent;
    if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) { break; }
    let s = textureSampleLevel(voxelRadiance, voxelSampler, uvw, lod);
    col = col + (1.0 - alpha) * s.rgb;
    alpha = alpha + (1.0 - alpha) * s.a;
    dist = dist + diameter * 0.5;
  }
  return vec4<f32>(col, alpha);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  let pixel = vec2<i32>(floor(input.position.xy));

  // World normal from the G-buffer; a<0.5 = no surface at this pixel.
  let n = textureLoad(normalTex, pixel, 0);
  if (n.a < 0.5) {
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }
  let N = normalize(n.rgb * 2.0 - 1.0);

  // Reconstruct world position from reverse-Z depth.
  let depth = textureLoad(depthTex, pixel, 0);
  let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) / uParams2.xy;
  let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth);
  let P = unproject(ndc);

  // Lift the cone origin off the surface to avoid self-sampling the originating voxel.
  let cellSize = uGridOrigin.w;
  let origin = P + N * (cellSize * 1.5 + uParams.x);

  // 6-cone cosine-weighted diffuse hemisphere gather around N.
  let basis = build_basis(N);
  let aperture = uParams.z;

  var acc = vec3<f32>(0.0);
  // Center cone along the normal, weight π/4.
  acc = acc + (PI / 4.0) * trace_cone(origin, N, aperture).rgb;

  // Ring of 5 side cones tilted 60° off the normal (local z = cos60 = 0.5, local xy
  // magnitude = sin60 ≈ 0.86602540), azimuth φ_i = i·(2π/5), weight 3π/20 each.
  for (var i = 0; i < 5; i = i + 1) {
    let phi = f32(i) * (2.0 * PI / 5.0);
    let local = vec3<f32>(0.86602540 * cos(phi), 0.86602540 * sin(phi), 0.5);
    let dir = normalize(basis * local);
    acc = acc + (3.0 * PI / 20.0) * trace_cone(origin, dir, aperture).rgb;
  }

  // Σweights = π/4 + 5·3π/20 = π → dividing gives the cosine-weighted average radiance.
  let irradiance = acc / PI;
  return vec4f(irradiance * uParams.w, 1.0);
}
`,
);
