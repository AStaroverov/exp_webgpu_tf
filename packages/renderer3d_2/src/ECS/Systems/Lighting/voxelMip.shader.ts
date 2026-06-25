import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";

// Voxel-radiance MIP downsample — one level per dispatch. Reads mip L of voxelRadiance
// (as a single-mip SAMPLED texture_3d via textureLoad) and writes mip L+1 (as a single-
// mip write-only StorageTexture). The 8 children of a destination voxel are combined with
// an ISOTROPIC, OPACITY-WEIGHTED downsample:
//   color = opacity-weighted MEAN radiance  = Σ(child.rgb · child.a) / Σ(child.a)
//   alpha = MEAN coverage                   = Σ(child.a) · 0.125          (stays in [0,1])
// Opacity-weighting the color means empty (a==0) children contribute nothing, so a voxel
// next to empty space does NOT darken toward black — this is what proves out as "no black
// holes" in the LOD debug view. This is the §5 isotropic recipe: a cleaner variant of
// maritim's (we use MEAN coverage, not Σa, so alpha never exceeds 1).

// COMPUTE-visibility group-0 uniform helper.
const uC = (name: string, type: string) =>
  new VariableMeta(name, VariableKind.Uniform, type, { visibility: GPUShaderStage.COMPUTE });

export const WORKGROUP = 4; // 4*4*4 = 64 threads/workgroup

export const shaderMeta = new ShaderMeta(
  {
    // ---- group 0 : uniforms + source mip (COMPUTE-only) ----
    // .xyz = DESTINATION mip dims (voxel counts of mip L+1), .w spare.
    mip: uC("uMip", `vec4<i32>`),
    // Source mip L of voxelRadiance, bound as a single-mip sampled 3D texture.
    src: new VariableMeta("src", VariableKind.Texture, `texture_3d<f32>`, {
      visibility: GPUShaderStage.COMPUTE,
      viewDimension: "3d",
      textureSampleType: "float",
    }),

    // ---- group 2 : destination mip L+1 (StorageTexture, write-only) ----
    dst: new VariableMeta("dst", VariableKind.StorageTexture, `texture_storage_3d<rgba16float, write>`, {
      visibility: GPUShaderStage.COMPUTE,
      viewDimension: "3d",
      storageTextureFormat: "rgba16float",
      storageTextureAccess: "write-only",
    }),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let coord = vec3<i32>(gid);
  // Dispatch is ceil-rounded; drop threads past the destination mip bounds.
  if (coord.x >= uMip.x || coord.y >= uMip.y || coord.z >= uMip.z) {
    return;
  }

  // The 8 children in the source mip occupy [base, base+1]^3.
  let base = coord * 2;
  var colAcc = vec3<f32>(0.0);
  var aAcc = 0.0;
  for (var dz = 0; dz < 2; dz = dz + 1) {
    for (var dy = 0; dy < 2; dy = dy + 1) {
      for (var dx = 0; dx < 2; dx = dx + 1) {
        let child = base + vec3<i32>(dx, dy, dz);
        let c = textureLoad(src, child, 0);
        colAcc = colAcc + c.rgb * c.a;
        aAcc = aAcc + c.a;
      }
    }
  }

  // Opacity-weighted mean color (empty children don't drag it to black); mean coverage.
  let outRgb = select(vec3<f32>(0.0), colAcc / aAcc, aAcc > 0.0);
  let outA = aAcc * 0.125;
  textureStore(dst, coord, vec4<f32>(outRgb, outA));
}
`,
);
