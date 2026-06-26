import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";

// Anisotropic VCT (Layer 6) — VOLUME pass. Downsamples each directional volume from level c to
// level c+1, each direction reading its OWN volume's 2×2×2 block with its OWN front-to-back
// ordering (jose-villegas aniso_mipmapvolume / rdinse PreIntegration). One dispatch per level c.
//
// Bindings: group 0 = {uDst Uniform, 6 sampled source volumes (level c, single-mip views)};
//           group 2 = the 6 write StorageTextures (level c+1 views). group 1 EMPTY.
// textureLoad(..., 0): each source view is a single-mip view (baseMipLevel=c, count=1), so the
// integer LOD arg is always 0 within that view. Each of the 6 per-direction blocks is wrapped in
// `{ }` so `let v0..v7` redeclares in its own scope (no duplicate-let error).

const uC = (name: string, type: string) =>
  new VariableMeta(name, VariableKind.Uniform, type, { visibility: GPUShaderStage.COMPUTE });

const srcTex = (name: string) =>
  new VariableMeta(name, VariableKind.Texture, `texture_3d<f32>`, {
    visibility: GPUShaderStage.COMPUTE,
    viewDimension: "3d",
    textureSampleType: "float",
  });

const storageTex = (name: string) =>
  new VariableMeta(name, VariableKind.StorageTexture, `texture_storage_3d<rgba16float, write>`, {
    visibility: GPUShaderStage.COMPUTE,
    viewDimension: "3d",
    storageTextureFormat: "rgba16float",
    storageTextureAccess: "write-only",
  });

export const WORKGROUP = 4;

export const shaderMeta = new ShaderMeta(
  {
    // ---- group 0 : uniform + the 6 sampled source volumes (level c) ----
    // .xyz = DESTINATION (level c+1) dims; .w unused.
    dst: uC("uDst", `vec4<i32>`),
    srcNegX: srcTex("srcNegX"),
    srcPosX: srcTex("srcPosX"),
    srcNegY: srcTex("srcNegY"),
    srcPosY: srcTex("srcPosY"),
    srcNegZ: srcTex("srcNegZ"),
    srcPosZ: srcTex("srcPosZ"),

    // ---- group 2 : the 6 directional level c+1 StorageTextures (write-only) ----
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

  {
    let v0 = textureLoad(srcNegX, base + vec3<i32>(1, 1, 1), 0);
    let v1 = textureLoad(srcNegX, base + vec3<i32>(1, 1, 0), 0);
    let v2 = textureLoad(srcNegX, base + vec3<i32>(1, 0, 1), 0);
    let v3 = textureLoad(srcNegX, base + vec3<i32>(1, 0, 0), 0);
    let v4 = textureLoad(srcNegX, base + vec3<i32>(0, 1, 1), 0);
    let v5 = textureLoad(srcNegX, base + vec3<i32>(0, 1, 0), 0);
    let v6 = textureLoad(srcNegX, base + vec3<i32>(0, 0, 1), 0);
    let v7 = textureLoad(srcNegX, base + vec3<i32>(0, 0, 0), 0);
    textureStore(dstNegX, dst, (v0 + v4 * (1.0 - v0.a) + v1 + v5 * (1.0 - v1.a) + v2 + v6 * (1.0 - v2.a) + v3 + v7 * (1.0 - v3.a)) * 0.25);
  }
  {
    let v0 = textureLoad(srcPosX, base + vec3<i32>(1, 1, 1), 0);
    let v1 = textureLoad(srcPosX, base + vec3<i32>(1, 1, 0), 0);
    let v2 = textureLoad(srcPosX, base + vec3<i32>(1, 0, 1), 0);
    let v3 = textureLoad(srcPosX, base + vec3<i32>(1, 0, 0), 0);
    let v4 = textureLoad(srcPosX, base + vec3<i32>(0, 1, 1), 0);
    let v5 = textureLoad(srcPosX, base + vec3<i32>(0, 1, 0), 0);
    let v6 = textureLoad(srcPosX, base + vec3<i32>(0, 0, 1), 0);
    let v7 = textureLoad(srcPosX, base + vec3<i32>(0, 0, 0), 0);
    textureStore(dstPosX, dst, (v4 + v0 * (1.0 - v4.a) + v5 + v1 * (1.0 - v5.a) + v6 + v2 * (1.0 - v6.a) + v7 + v3 * (1.0 - v7.a)) * 0.25);
  }
  {
    let v0 = textureLoad(srcNegY, base + vec3<i32>(1, 1, 1), 0);
    let v1 = textureLoad(srcNegY, base + vec3<i32>(1, 1, 0), 0);
    let v2 = textureLoad(srcNegY, base + vec3<i32>(1, 0, 1), 0);
    let v3 = textureLoad(srcNegY, base + vec3<i32>(1, 0, 0), 0);
    let v4 = textureLoad(srcNegY, base + vec3<i32>(0, 1, 1), 0);
    let v5 = textureLoad(srcNegY, base + vec3<i32>(0, 1, 0), 0);
    let v6 = textureLoad(srcNegY, base + vec3<i32>(0, 0, 1), 0);
    let v7 = textureLoad(srcNegY, base + vec3<i32>(0, 0, 0), 0);
    textureStore(dstNegY, dst, (v0 + v2 * (1.0 - v0.a) + v1 + v3 * (1.0 - v1.a) + v5 + v7 * (1.0 - v5.a) + v4 + v6 * (1.0 - v4.a)) * 0.25);
  }
  {
    let v0 = textureLoad(srcPosY, base + vec3<i32>(1, 1, 1), 0);
    let v1 = textureLoad(srcPosY, base + vec3<i32>(1, 1, 0), 0);
    let v2 = textureLoad(srcPosY, base + vec3<i32>(1, 0, 1), 0);
    let v3 = textureLoad(srcPosY, base + vec3<i32>(1, 0, 0), 0);
    let v4 = textureLoad(srcPosY, base + vec3<i32>(0, 1, 1), 0);
    let v5 = textureLoad(srcPosY, base + vec3<i32>(0, 1, 0), 0);
    let v6 = textureLoad(srcPosY, base + vec3<i32>(0, 0, 1), 0);
    let v7 = textureLoad(srcPosY, base + vec3<i32>(0, 0, 0), 0);
    textureStore(dstPosY, dst, (v2 + v0 * (1.0 - v2.a) + v3 + v1 * (1.0 - v3.a) + v7 + v5 * (1.0 - v7.a) + v6 + v4 * (1.0 - v6.a)) * 0.25);
  }
  {
    let v0 = textureLoad(srcNegZ, base + vec3<i32>(1, 1, 1), 0);
    let v1 = textureLoad(srcNegZ, base + vec3<i32>(1, 1, 0), 0);
    let v2 = textureLoad(srcNegZ, base + vec3<i32>(1, 0, 1), 0);
    let v3 = textureLoad(srcNegZ, base + vec3<i32>(1, 0, 0), 0);
    let v4 = textureLoad(srcNegZ, base + vec3<i32>(0, 1, 1), 0);
    let v5 = textureLoad(srcNegZ, base + vec3<i32>(0, 1, 0), 0);
    let v6 = textureLoad(srcNegZ, base + vec3<i32>(0, 0, 1), 0);
    let v7 = textureLoad(srcNegZ, base + vec3<i32>(0, 0, 0), 0);
    textureStore(dstNegZ, dst, (v0 + v1 * (1.0 - v0.a) + v2 + v3 * (1.0 - v2.a) + v4 + v5 * (1.0 - v4.a) + v6 + v7 * (1.0 - v6.a)) * 0.25);
  }
  {
    let v0 = textureLoad(srcPosZ, base + vec3<i32>(1, 1, 1), 0);
    let v1 = textureLoad(srcPosZ, base + vec3<i32>(1, 1, 0), 0);
    let v2 = textureLoad(srcPosZ, base + vec3<i32>(1, 0, 1), 0);
    let v3 = textureLoad(srcPosZ, base + vec3<i32>(1, 0, 0), 0);
    let v4 = textureLoad(srcPosZ, base + vec3<i32>(0, 1, 1), 0);
    let v5 = textureLoad(srcPosZ, base + vec3<i32>(0, 1, 0), 0);
    let v6 = textureLoad(srcPosZ, base + vec3<i32>(0, 0, 1), 0);
    let v7 = textureLoad(srcPosZ, base + vec3<i32>(0, 0, 0), 0);
    textureStore(dstPosZ, dst, (v1 + v0 * (1.0 - v1.a) + v3 + v2 * (1.0 - v3.a) + v5 + v4 * (1.0 - v5.a) + v7 + v6 * (1.0 - v7.a)) * 0.25);
  }
}
`,
);
