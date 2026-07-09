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

// The 2×2×2 sub-voxel offsets, in load order v0..v7. Index i encodes the offset bits:
// offset = (1 - ((i>>2)&1), 1 - ((i>>1)&1), 1 - (i&1)) → v0=(1,1,1) .. v7=(0,0,0).
const OFFSETS: [number, number, number][] = [
  [1, 1, 1],
  [1, 1, 0],
  [1, 0, 1],
  [1, 0, 0],
  [0, 1, 1],
  [0, 1, 0],
  [0, 0, 1],
  [0, 0, 0],
];

// One block per direction. `pairs` are the front-to-back (near, far) sample pairs along that
// direction's axis, in the ORIGINAL accumulation order (kept exact so the summation is FP-identical
// to the old hand-written blocks). The near voxel occludes the far one: near + far·(1 − near.a).
const DIRS: { name: string; pairs: [number, number][] }[] = [
  {
    name: "NegX",
    pairs: [
      [0, 4],
      [1, 5],
      [2, 6],
      [3, 7],
    ],
  },
  {
    name: "PosX",
    pairs: [
      [4, 0],
      [5, 1],
      [6, 2],
      [7, 3],
    ],
  },
  {
    name: "NegY",
    pairs: [
      [0, 2],
      [1, 3],
      [5, 7],
      [4, 6],
    ],
  },
  {
    name: "PosY",
    pairs: [
      [2, 0],
      [3, 1],
      [7, 5],
      [6, 4],
    ],
  },
  {
    name: "NegZ",
    pairs: [
      [0, 1],
      [2, 3],
      [4, 5],
      [6, 7],
    ],
  },
  {
    name: "PosZ",
    pairs: [
      [1, 0],
      [3, 2],
      [5, 4],
      [7, 6],
    ],
  },
];

// Emit one per-direction block: 8 loads from src{Dir} + the front-to-back store into dst{Dir}.
// Each block is wrapped in `{ }` so `let v0..v7` gets its own scope (no duplicate-let across blocks).
const anisoBlock = ({ name, pairs }: (typeof DIRS)[number]) => {
  const loads = OFFSETS.map(
    (o, i) =>
      `    let v${i} = textureLoad(src${name}, base + vec3<i32>(${o[0]}, ${o[1]}, ${o[2]}), 0);`,
  ).join("\n");
  const acc = pairs.map(([n, f]) => `v${n} + v${f} * (1.0 - v${n}.a)`).join(" + ");
  return `  {\n${loads}\n    textureStore(dst${name}, dst, (${acc}) * 0.25);\n  }`;
};

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

${DIRS.map(anisoBlock).join("\n")}
}
`,
);
