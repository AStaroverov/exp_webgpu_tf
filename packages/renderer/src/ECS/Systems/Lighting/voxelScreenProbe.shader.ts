import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";
import { VoxelBakedConfig } from "./voxelConfig.ts";

// VCT — SCREEN-SPACE PROBE gather (one compute pass, one thread per screen-probe). This is the
// A/B alternative to the WORLD SH-L1 probe volume (voxelProbe): instead of probes on a fixed 3D
// grid through the whole box, probes are anchored to the VISIBLE SURFACE — one probe per 16×16
// full-canvas-pixel tile, placed at the tile's representative G-buffer pixel. Each probe traces
// a full SPHERE of fill cones through the voxelRadiance pyramid, projects the gathered radiance
// onto SH-L1 (4 coeffs / channel), and stores it into three rgba16float 2D textures (one per
// color channel, .xyzw = L00,L1m1,L10,L11) — SAME layout + scale as the world volume, so the
// cone-pass fill term switches between the two by one uniform bit.
//
// WHY SCREEN-SPACE: probe density follows the camera (no cells wasted on off-screen / empty
// space), and the placement is exact on the visible surface, so the fill resolves finer where
// the eye looks. The trade-off (disocclusion / silhouette gaps) is covered by the strict
// bilateral resolve + a clean fallback to the always-defined world volume — see voxelCone.
//
// NO temporal accumulation, NO history: probes are regenerated from scratch each frame from the
// live voxelRadiance. Noise is a non-issue because the gather is VCT cone tracing (prefiltered),
// not ray tracing. The gather integration is a full-sphere Fibonacci cone set projected onto SH-L1
// (the standard low-frequency diffuse-bounce representation); the only thing "screen" about it is
// WHERE the probe sits (surface-anchored screen tile), not the gather math.
//
// STORAGE = RAW SH RADIANCE coefficients (solid-angle weighted, no cosine lobe). The cosine
// (irradiance) convolution is applied at RECONSTRUCTION time in the cone shader (sh_avg_radiance).
// The 4th storage slot (WebGPU default caps a compute stage at 4 storage textures) holds the probe's
// representative full-res pixel coord + validity (screenProbePix); the resolve reconstructs the
// probe's world position + normal from the G-buffer at that pixel.

// COMPUTE-visibility group-0 uniform helper.
const uC = (name: string, type: string) =>
  new VariableMeta(name, VariableKind.Uniform, type, { visibility: GPUShaderStage.COMPUTE });

export const WORKGROUP = 8; // 8*8 = 64 threads/workgroup over the 2D probe grid

export function createScreenProbeShaderMeta(cfg: VoxelBakedConfig) {
  return new ShaderMeta(
  {
    // ---- group 0 : uniforms (COMPUTE-only) ----
    // .xyz = world min corner of the grid box, .w = cellSize (world units per voxel). SAME box
    // as the cone/voxelize grid → the cone march reads it via uGridOrigin/uGridDims.
    gridOrigin: uC("uGridOrigin", `vec4<f32>`),
    // .xyz = voxel counts per axis (defines the world extent together with cellSize), .w unused.
    gridDims: uC("uGridDims", `vec4<i32>`),
    // inverse(viewProjMatrix) (reverse-Z), column-major — reconstructs the probe's world position
    // from the G-buffer depth at its representative pixel (uploaded each frame in screenProbe()).
    invViewProj: uC("uInvViewProj", `mat4x4<f32>`),
    // .x = canvas width (px), .y = canvas height (px), .z = SCREEN_PROBE_TILE (full-res px / probe),
    // .w = edgeAware (0 = tile-center placement, 1 = snap the probe onto the NEAREST valid surface
    // in the tile so a thin FOREGROUND feature that misses the center still gets a probe on it).
    screenParams: uC("screenParams", `vec4<f32>`),

    // ---- group 0 : G-buffer + voxelRadiance pyramid (Texture/Sampler => @group(0)) ----
    // Declared as Texture/Sampler so they land in group 0 alongside the uniforms (mapKindToGroup),
    // keeping group 2 free for the SH + pixel storage outputs (StorageTexture => group 2).
    depthTex: new VariableMeta("depthTex", VariableKind.Texture, `texture_depth_2d`, {
      visibility: GPUShaderStage.COMPUTE,
      textureSampleType: "depth",
    }),
    normalTex: new VariableMeta("normalTex", VariableKind.Texture, `texture_2d<f32>`, {
      visibility: GPUShaderStage.COMPUTE,
      textureSampleType: "float",
    }),
    voxelRadiance: new VariableMeta("voxelRadiance", VariableKind.Texture, `texture_3d<f32>`, {
      visibility: GPUShaderStage.COMPUTE,
      viewDimension: "3d",
      textureSampleType: "float",
    }),
    voxelSampler: new VariableMeta("voxelSampler", VariableKind.Sampler, `sampler`, {
      visibility: GPUShaderStage.COMPUTE,
    }),

    // ---- group 2 : outputs (StorageTexture, write-only, 4 total = the WebGPU default cap) ----
    // The 3 SH-L1 channel textures (.xyzw = the 4 SH-L1 coeffs L00,L1m1,L10,L11 of that channel's
    // RADIANCE) + the probe-geometry texture (.xy = representative full-res pixel, .z = validity).
    screenShR: new VariableMeta("screenShR", VariableKind.StorageTexture, `texture_storage_2d<rgba16float, write>`, {
      visibility: GPUShaderStage.COMPUTE,
      viewDimension: "2d",
      storageTextureFormat: "rgba16float",
      storageTextureAccess: "write-only",
    }),
    screenShG: new VariableMeta("screenShG", VariableKind.StorageTexture, `texture_storage_2d<rgba16float, write>`, {
      visibility: GPUShaderStage.COMPUTE,
      viewDimension: "2d",
      storageTextureFormat: "rgba16float",
      storageTextureAccess: "write-only",
    }),
    screenShB: new VariableMeta("screenShB", VariableKind.StorageTexture, `texture_storage_2d<rgba16float, write>`, {
      visibility: GPUShaderStage.COMPUTE,
      viewDimension: "2d",
      storageTextureFormat: "rgba16float",
      storageTextureAccess: "write-only",
    }),
    // .xy = representative full-res pixel coord, .z = validity (1 surface / 0 no surface). rgba32float
    // (point-loaded in the resolve) so the pixel coords survive without precision loss.
    screenProbePix: new VariableMeta("screenProbePix", VariableKind.StorageTexture, `texture_storage_2d<rgba32float, write>`, {
      visibility: GPUShaderStage.COMPUTE,
      viewDimension: "2d",
      storageTextureFormat: "rgba32float",
      storageTextureAccess: "write-only",
    }),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
const PI: f32 = 3.14159265359;
const CONES_PER_PROBE: i32 = ${cfg.conesPerProbe};
const MAX_DIST: f32 = ${cfg.maxDist};
const APERTURE: f32 = ${cfg.aperture};
const NORMAL_BIAS: f32 = ${cfg.normalBias};

// One fill cone marched from origin along dir through the voxelRadiance pyramid: premultiplied
// front-to-back "over" integration (diameter grows with distance, LOD = log2(diameter/voxelSize),
// step floored at reach/64, early-out on full opacity / past reach / outside the box) — the same
// integration the per-pixel cone gather uses, so the fill matches the aimed/AO cones' scale.
fn trace_probe_cone(origin: vec3<f32>, dir: vec3<f32>, aperture: f32, reach: f32) -> vec4<f32> {
  var col = vec3<f32>(0.0);
  var alpha = 0.0;
  let voxelSize = uGridOrigin.w;
  let gridMin = uGridOrigin.xyz;
  let extent = vec3<f32>(uGridDims.xyz) * voxelSize;
  let stepFloor = reach / 64.0;
  var dist = voxelSize;
  for (var i = 0; i < 64; i = i + 1) {
    if (alpha >= 1.0 || dist > reach) { break; }
    let diameter = max(voxelSize, 2.0 * aperture * dist);
    let lod = log2(diameter / voxelSize);
    let wp = origin + dir * dist;
    let uvw = (wp - gridMin) / extent;
    if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) { break; }
    let s = textureSampleLevel(voxelRadiance, voxelSampler, uvw, lod);
    col = col + (1.0 - alpha) * s.rgb;
    alpha = alpha + (1.0 - alpha) * s.a;
    dist = dist + max(diameter * 0.5, stepFloor);
  }
  return vec4<f32>(col, alpha);
}

// Unproject an NDC point (z reverse-Z) to world space. Copied from voxelCone.shader.ts.
fn unproject(ndc: vec3<f32>) -> vec3<f32> {
  let w = uInvViewProj * vec4<f32>(ndc, 1.0);
  return w.xyz / w.w;
}

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let coord = vec2<i32>(gid.xy);
  let tile = i32(screenParams.z);
  // Probe-grid dims = ceil(canvas / tile). Dispatch is ceil-rounded; drop threads past them.
  let gw = i32(ceil(screenParams.x / screenParams.z));
  let gh = i32(ceil(screenParams.y / screenParams.z));
  if (coord.x >= gw || coord.y >= gh) {
    return;
  }

  // Representative full-res pixel = tile center + a STABLE per-tile spatial jitter (hashed from
  // coord → the SAME every frame, so the probe placement never flickers; mirrors the cone pass's
  // spatial jitter, NEVER temporal). Kept within ±tile/4 of the center; clamped into the canvas.
  let h = fract(52.9829189 * fract(dot(vec2<f32>(coord), vec2<f32>(0.06711056, 0.00583715))));
  let jit = vec2<i32>((vec2<f32>(h, fract(h * 1.61803399)) - vec2<f32>(0.5)) * screenParams.z * 0.5);
  let center = coord * tile + vec2<i32>(tile / 2) + jit;
  let dimsI = vec2<i32>(screenParams.xy) - vec2<i32>(1);
  var full = min(center, dimsI);

  // EDGE-AWARE placement (screenParams.w): instead of trusting the tile center — which on a
  // silhouette tile usually lands on the BACKGROUND and leaves a thin foreground feature with no
  // probe — scan a coarse grid of the tile and snap the probe onto the NEAREST valid surface
  // (reverse-Z: near = 1 → the largest depth). On a flat tile every tap has ~the same depth, so the
  // pick reduces to ~the center (self-adaptive); on a depth discontinuity it favours the foreground,
  // so thin near objects get a probe that represents THEM. Pure gather-side — the resolve is
  // unchanged (it already reconstructs P/N from whatever pixel screenProbePix stores).
  if (screenParams.w > 0.5) {
    let base = coord * tile;
    let step = max(1, tile / 4);   // ~4x4 taps across the tile
    var bestDepth = -1.0;          // reverse-Z depth of the nearest valid tap (-1 = none yet)
    for (var sy = 0; sy < tile; sy = sy + step) {
      for (var sx = 0; sx < tile; sx = sx + step) {
        let p = min(base + vec2<i32>(sx, sy), dimsI);
        if (textureLoad(normalTex, p, 0).a < 0.5) { continue; }  // no surface (sky) → skip
        let d = textureLoad(depthTex, p, 0);
        if (d > bestDepth) { bestDepth = d; full = p; }
      }
    }
    // bestDepth < 0 → the whole tile is empty; full stays the clamped center and the n.a<0.5
    // check below marks the probe invalid, exactly as the center-placement path would.
  }

  // No surface at this pixel → mark the probe invalid (the resolve leans on neighbors) and zero its
  // SH. n.a<0.5 = the G-buffer has no geometry here.
  let n = textureLoad(normalTex, full, 0);
  if (n.a < 0.5) {
    textureStore(screenShR, coord, vec4<f32>(0.0));
    textureStore(screenShG, coord, vec4<f32>(0.0));
    textureStore(screenShB, coord, vec4<f32>(0.0));
    textureStore(screenProbePix, coord, vec4<f32>(f32(full.x), f32(full.y), 0.0, 0.0));
    return;
  }

  // Reconstruct the probe anchor P + normal N from the G-buffer, then lift the cone origin off
  // the surface (SAME lift as voxelCone's per-pixel origin) to avoid self-sampling the surface voxel.
  let N = normalize(n.rgb * 2.0 - 1.0);
  let depth = textureLoad(depthTex, full, 0);
  let uv = (vec2<f32>(full) + vec2<f32>(0.5)) / screenParams.xy;
  let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth);
  let P = unproject(ndc);
  let origin = P + N * (uGridOrigin.w * 1.5 + NORMAL_BIAS);

  let C = CONES_PER_PROBE;
  let reach = MAX_DIST;
  let aperture = APERTURE;

  // SH-L1 radiance accumulators, per color channel (.xyzw = L00, L1m1, L10, L11).
  var cR = vec4<f32>(0.0);
  var cG = vec4<f32>(0.0);
  var cB = vec4<f32>(0.0);

  // Trace C fill cones over the FULL SPHERE (Fibonacci sphere), EXACTLY as voxelProbe.main: same
  // direction set, same dw = 4*PI/C solid-angle weight, same real SH-L1 basis → the two fill
  // sources are numerically comparable by construction. Cones aimed into the surface simply return
  // the lit surface voxel (a legitimate upward bounce); the origin lift avoids self-sampling.
  let dw = 4.0 * PI / f32(C);
  for (var i = 0; i < C; i = i + 1) {
    let cosT = 1.0 - 2.0 * (f32(i) + 0.5) / f32(C);
    let sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
    let phi = f32(i) * 2.39996323;            // golden angle
    let dir = vec3<f32>(sinT * cos(phi), sinT * sin(phi), cosT);

    let r = trace_probe_cone(origin, dir, aperture, reach);

    // Real SH-L1 basis evaluated at dir.
    let Y00 = 0.282095;
    let Y1m1 = 0.488603 * dir.y;
    let Y10 = 0.488603 * dir.z;
    let Y11 = 0.488603 * dir.x;
    let basis = vec4<f32>(Y00, Y1m1, Y10, Y11) * dw;
    cR = cR + r.r * basis;
    cG = cG + r.g * basis;
    cB = cB + r.b * basis;
  }

  textureStore(screenShR, coord, cR);
  textureStore(screenShG, coord, cG);
  textureStore(screenShB, coord, cB);
  // Valid probe: store the representative full-res pixel + validity 1 for the bilateral resolve.
  textureStore(screenProbePix, coord, vec4<f32>(f32(full.x), f32(full.y), 1.0, 0.0));
}
`,
  );
}
