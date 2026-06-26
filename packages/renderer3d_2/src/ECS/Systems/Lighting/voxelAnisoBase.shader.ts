import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";

// Anisotropic VCT (Layer 6) — BASE pass. Builds the level-0 of the 6 directional radiance
// volumes from the ISOTROPIC voxelRadiance mip 0 (full res, PREMULTIPLIED rgb=radiance·coverage,
// a=coverage). Each directional level-0 voxel covers the 2×2×2 block of iso mip 0 and accumulates
// it FRONT-TO-BACK along that direction's axis (rdinse/jose-villegas pre-integration), so the
// near occluder masks the far voxel it shadows → DIRECTION-correct occlusion (the anti-leak).
//
// The 8-child fetch order (rdinse): i0=(1,1,1) i1=(1,1,0) i2=(1,0,1) i3=(1,0,0)
//                                   i4=(0,1,1) i5=(0,1,0) i6=(0,0,1) i7=(0,0,0).
// Direction convention: 0=-X,1=+X,2=-Y,3=+Y,4=-Z,5=+Z (matches the sampler's visibleFace).
// Premultiplied source → the "over" formula (near + far·(1-near.a)) applies directly.
//
// Bindings: group 0 = {uDst Uniform, srcIso Texture (iso mip 0, single-mip sampled view)};
//           group 2 = the 6 write StorageTextures (directional level 0). group 1 EMPTY (bind an
//           empty group there, like voxelMip). Reads iso mip 0 ONLY (uDst.w = 0): reading an
//           already-isotropically-averaged mip would defeat the anti-leak.

// COMPUTE-visibility group-0 uniform helper (mirrors voxelMip.shader.ts).
const uC = (name: string, type: string) =>
  new VariableMeta(name, VariableKind.Uniform, type, { visibility: GPUShaderStage.COMPUTE });

const storageTex = (name: string) =>
  new VariableMeta(name, VariableKind.StorageTexture, `texture_storage_3d<rgba16float, write>`, {
    visibility: GPUShaderStage.COMPUTE,
    viewDimension: "3d",
    storageTextureFormat: "rgba16float",
    storageTextureAccess: "write-only",
  });

export const WORKGROUP = 4; // 4*4*4 = 64 threads/workgroup (rdinse's measured optimum)

export const shaderMeta = new ShaderMeta(
  {
    // ---- group 0 : uniform + iso source mip (COMPUTE-only) ----
    // .xyz = DESTINATION (directional level-0) dims; .w = iso source mip level (always 0).
    dst: uC("uDst", `vec4<i32>`),
    // Iso voxelRadiance, bound as a single-mip (mip 0) sampled 3D texture.
    srcIso: new VariableMeta("srcIso", VariableKind.Texture, `texture_3d<f32>`, {
      visibility: GPUShaderStage.COMPUTE,
      viewDimension: "3d",
      textureSampleType: "float",
    }),

    // ---- group 2 : the 6 directional level-0 StorageTextures (write-only) ----
    dstNegX: storageTex("dstNegX"),
    dstPosX: storageTex("dstPosX"),
    dstNegY: storageTex("dstNegY"),
    dstPosY: storageTex("dstPosY"),
    dstNegZ: storageTex("dstNegZ"),
    dstPosZ: storageTex("dstPosZ"),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dst = vec3<i32>(gid);
  if (dst.x >= uDst.x || dst.y >= uDst.y || dst.z >= uDst.z) { return; }

  let base = dst * 2;
  let srcLod = uDst.w;
  let v0 = textureLoad(srcIso, base + vec3<i32>(1, 1, 1), srcLod);
  let v1 = textureLoad(srcIso, base + vec3<i32>(1, 1, 0), srcLod);
  let v2 = textureLoad(srcIso, base + vec3<i32>(1, 0, 1), srcLod);
  let v3 = textureLoad(srcIso, base + vec3<i32>(1, 0, 0), srcLod);
  let v4 = textureLoad(srcIso, base + vec3<i32>(0, 1, 1), srcLod);
  let v5 = textureLoad(srcIso, base + vec3<i32>(0, 1, 0), srcLod);
  let v6 = textureLoad(srcIso, base + vec3<i32>(0, 0, 1), srcLod);
  let v7 = textureLoad(srcIso, base + vec3<i32>(0, 0, 0), srcLod);

  textureStore(dstNegX, dst, (v0 + v4 * (1.0 - v0.a) + v1 + v5 * (1.0 - v1.a) + v2 + v6 * (1.0 - v2.a) + v3 + v7 * (1.0 - v3.a)) * 0.25);
  textureStore(dstPosX, dst, (v4 + v0 * (1.0 - v4.a) + v5 + v1 * (1.0 - v5.a) + v6 + v2 * (1.0 - v6.a) + v7 + v3 * (1.0 - v7.a)) * 0.25);
  textureStore(dstNegY, dst, (v0 + v2 * (1.0 - v0.a) + v1 + v3 * (1.0 - v1.a) + v5 + v7 * (1.0 - v5.a) + v4 + v6 * (1.0 - v4.a)) * 0.25);
  textureStore(dstPosY, dst, (v2 + v0 * (1.0 - v2.a) + v3 + v1 * (1.0 - v3.a) + v7 + v5 * (1.0 - v7.a) + v6 + v4 * (1.0 - v6.a)) * 0.25);
  textureStore(dstNegZ, dst, (v0 + v1 * (1.0 - v0.a) + v2 + v3 * (1.0 - v2.a) + v4 + v5 * (1.0 - v4.a) + v6 + v7 * (1.0 - v6.a)) * 0.25);
  textureStore(dstPosZ, dst, (v1 + v0 * (1.0 - v1.a) + v3 + v2 * (1.0 - v3.a) + v5 + v4 * (1.0 - v5.a) + v7 + v6 * (1.0 - v7.a)) * 0.25);
}
`,
);
