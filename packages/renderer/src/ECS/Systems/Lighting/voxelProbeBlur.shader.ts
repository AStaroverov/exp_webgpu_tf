import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";
import { VoxelBakedConfig } from "./voxelConfig.ts";

// VCT — PROBE-VOLUME BLUR (SEPARABLE 3D Gaussian over the SH-L1 irradiance volume). Smooths the
// coarse probe bounce so a MOVING source's fill stops "stepping" by probe cells, written through a
// SECOND SH set (ping-pong) so the cone pass samples the blurred result.
//
// WHY A BLUR (and why it's cheap): the probe grid is coarse (probeDims << gridDims, ~2-unit
// spacing). Its trilinear fetch is C0-continuous, but as a source MOVES the per-probe values jump
// vs. their neighbours and the bounce steps by cells. Raising probe resolution fixes it at
// O(probes·cones·steps) — the probe TRACE is the expensive part. A blur fixes the SAME artifact
// without any extra cones/marching.
//
// WHY SEPARABLE: a Gaussian factorizes G(x,y,z)=g(x)·g(y)·g(z), so three 1D passes (X then Y then
// Z) produce the IDENTICAL result as a dense (2R+1)³ kernel but cost O(R) taps per pass instead of
// O(R³) total — the difference between 27 and 729 taps at R=4. Each pass ping-pongs between the two
// SH sets: X: A→B, Y: B→A, Z: A→B, so the FINAL blurred volume lands in set B (probeTexturesBlur),
// which the cone pass reads. The axis is BAKED per entry point (blur_x/_y/_z) → no per-pass uniform.
//
// VALID because SH-L1 is LINEAR: blurring the coefficients spatially == blurring the reconstructed
// radiance field (reconstruction is linear in the coeffs). Purely SPATIAL → no temporal history,
// no ghosting (consistent with the probe pass). R is BAKED (config.probeBlurRadius); R=0 → a single
// tap = passthrough copy. textureLoad (integer coords, no sampler — exact per-probe fetch).

// COMPUTE-visibility group-0 uniform helper (mirrors voxelProbe.shader.ts).
const uC = (name: string, type: string) =>
  new VariableMeta(name, VariableKind.Uniform, type, { visibility: GPUShaderStage.COMPUTE });

export const WORKGROUP = 4; // 4*4*4 = 64 threads/workgroup over the 3D probe grid

export function createProbeBlurShaderMeta(cfg: VoxelBakedConfig) {
  return new ShaderMeta(
    {
      // ---- group 0 : uniforms + source SH textures (Texture => @group(0)) ----
      // .xyz = probe counts per axis (bounds + clamp), .w unused.
      probeDims: uC("uProbeDims", `vec4<i32>`),
      // The SH-L1 volume to blur for THIS pass (one texture per color channel; .xyzw = the 4
      // coeffs). Read via textureLoad (integer coords, LOD 0) — no sampler.
      srcR: new VariableMeta("srcR", VariableKind.Texture, `texture_3d<f32>`, {
        visibility: GPUShaderStage.COMPUTE,
        viewDimension: "3d",
        textureSampleType: "float",
      }),
      srcG: new VariableMeta("srcG", VariableKind.Texture, `texture_3d<f32>`, {
        visibility: GPUShaderStage.COMPUTE,
        viewDimension: "3d",
        textureSampleType: "float",
      }),
      srcB: new VariableMeta("srcB", VariableKind.Texture, `texture_3d<f32>`, {
        visibility: GPUShaderStage.COMPUTE,
        viewDimension: "3d",
        textureSampleType: "float",
      }),

      // ---- group 2 : destination SH outputs (StorageTexture, write-only) ----
      dstR: new VariableMeta("dstR", VariableKind.StorageTexture, `texture_storage_3d<rgba16float, write>`, {
        visibility: GPUShaderStage.COMPUTE,
        viewDimension: "3d",
        storageTextureFormat: "rgba16float",
        storageTextureAccess: "write-only",
      }),
      dstG: new VariableMeta("dstG", VariableKind.StorageTexture, `texture_storage_3d<rgba16float, write>`, {
        visibility: GPUShaderStage.COMPUTE,
        viewDimension: "3d",
        storageTextureFormat: "rgba16float",
        storageTextureAccess: "write-only",
      }),
      dstB: new VariableMeta("dstB", VariableKind.StorageTexture, `texture_storage_3d<rgba16float, write>`, {
        visibility: GPUShaderStage.COMPUTE,
        viewDimension: "3d",
        storageTextureFormat: "rgba16float",
        storageTextureAccess: "write-only",
      }),
    },
    {},
    // language=WGSL
    wgsl /* wgsl */ `
const R: i32 = ${cfg.probeBlurRadius};

// 1D Gaussian weight; sigma = R/2 (a moderate bell that still tapers within the radius). R=0 → the
// loop runs only i=0 and gw(0)=1, so the pass is an exact copy (passthrough).
fn gw(i: i32) -> f32 {
  let sigma = max(0.5, f32(R) * 0.5);
  return exp(-f32(i * i) / (2.0 * sigma * sigma));
}

// One separable 1D blur along the given axis (a unit step vector). Reads src*, writes dst*.
fn blur_1d(coord: vec3<i32>, axis: vec3<i32>) {
  // Dispatch is ceil-rounded; drop threads past the probe-grid bounds.
  if (coord.x >= uProbeDims.x || coord.y >= uProbeDims.y || coord.z >= uProbeDims.z) {
    return;
  }
  var aR = vec4<f32>(0.0);
  var aG = vec4<f32>(0.0);
  var aB = vec4<f32>(0.0);
  var wsum = 0.0;
  let hi = uProbeDims.xyz - vec3<i32>(1);
  for (var i = -R; i <= R; i = i + 1) {
    // Clamp to edge so border probes don't darken (no wrap, no out-of-bounds).
    let c = clamp(coord + axis * i, vec3<i32>(0), hi);
    let w = gw(i);
    aR = aR + textureLoad(srcR, c, 0) * w;
    aG = aG + textureLoad(srcG, c, 0) * w;
    aB = aB + textureLoad(srcB, c, 0) * w;
    wsum = wsum + w;
  }
  let inv = 1.0 / wsum;
  textureStore(dstR, coord, aR * inv);
  textureStore(dstG, coord, aG * inv);
  textureStore(dstB, coord, aB * inv);
}

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP}, ${WORKGROUP})
fn blur_x(@builtin(global_invocation_id) gid: vec3<u32>) {
  blur_1d(vec3<i32>(gid), vec3<i32>(1, 0, 0));
}

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP}, ${WORKGROUP})
fn blur_y(@builtin(global_invocation_id) gid: vec3<u32>) {
  blur_1d(vec3<i32>(gid), vec3<i32>(0, 1, 0));
}

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP}, ${WORKGROUP})
fn blur_z(@builtin(global_invocation_id) gid: vec3<u32>) {
  blur_1d(vec3<i32>(gid), vec3<i32>(0, 0, 1));
}
`,
  );
}
