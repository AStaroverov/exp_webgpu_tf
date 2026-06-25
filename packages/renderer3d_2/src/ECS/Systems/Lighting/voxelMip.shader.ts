import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";

// Voxel-radiance MIP downsample — one level per dispatch. Reads mip L of voxelRadiance
// (as a single-mip SAMPLED texture_3d via textureLoad) and writes mip L+1 (as a single-
// mip write-only StorageTexture). The 8 children of a destination voxel are combined with
// an ISOTROPIC, PREMULTIPLIED-ALPHA box average:
//   color = MEAN premultiplied radiance     = Σ(child.rgb) · 0.125   (child.rgb is premult)
//   alpha = MEAN coverage                    = Σ(child.a) · 0.125    (stays in [0,1])
// voxelRadiance is stored PREMULTIPLIED (mip 0 = radiance·coverage, coverage a∈{0,1}; empty
// voxels are 0), and the cone trace composites front-to-back with a premultiplied over-
// operator (col += (1-α)·rgb). So the downsample MUST keep things premultiplied — a plain
// box mean, NOT an opacity-weighted Σ(rgb·a)/Σa. Opacity-weighting un-premultiplies the
// color, which makes a sub-texel emitter paint its FULL colour across the whole coarse
// texel → "light drifts / shows up behind objects at high LOD". A premultiplied mean fades
// the emitter with its coverage (energy-conserving) while fully-solid regions (all a=1) do
// not darken (Σrgb·0.125 = rgb). Matches the energy-conserving downsample in sfreed141 /
// rdinse (front-to-back) rather than maritim's opacity-weighted form.

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
    dst: new VariableMeta(
      "dst",
      VariableKind.StorageTexture,
      `texture_storage_3d<rgba16float, write>`,
      {
        visibility: GPUShaderStage.COMPUTE,
        viewDimension: "3d",
        storageTextureFormat: "rgba16float",
        storageTextureAccess: "write-only",
      },
    ),
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

  // The 8 children in the source mip occupy [base, base+1]^3. Accumulate the three pieces
  // the variants below need: Σrgb, Σ(rgb·a), Σa.
  let base = coord * 2;
  var sumRgb = vec3<f32>(0.0);  // Σ child.rgb (already premultiplied: mip0 = radiance·a, a∈{0,1})
  var sumRgbA = vec3<f32>(0.0); // Σ child.rgb·child.a
  var sumA = 0.0;               // Σ child.a (coverage)
  for (var dz = 0; dz < 2; dz = dz + 1) {
    for (var dy = 0; dy < 2; dy = dy + 1) {
      for (var dx = 0; dx < 2; dx = dx + 1) {
        let child = base + vec3<i32>(dx, dy, dz);
        let c = textureLoad(src, child, 0);
        sumRgb = sumRgb + c.rgb;
        sumRgbA = sumRgbA + c.rgb * c.a;
        sumA = sumA + c.a;
      }
    }
  }
  let outA = sumA * 0.125; // mean coverage — same for all variants

  // ===== DOWNSAMPLE COLOR VARIANTS — keep exactly ONE 'outRgb' uncommented; test iteratively.
  // (sumRgbA is only read by B/C; an unused local is legal WGSL, so leave it accumulated.)

  // A) PREMULTIPLIED MEAN  (recommended; energy-conserving, consistent with the cone's
  //    front-to-back over-operator). Emitters FADE as they spread; solid regions don't darken.
  //    Like sfreed141 / rdinse. Dimmer-but-correct; expect a soft, even indirect.
  let outRgb = sumRgb * 0.125;

  // B) OPACITY-WEIGHTED MEAN  (original Layer-1 / maritim). Full colour at low coverage → a
  //    sub-texel emitter paints its FULL colour across the whole coarse texel: BRIGHTEST, most
  //    "alive", but light spreads / drifts behind objects at high LOD.
  // let outRgb = select(vec3<f32>(0.0), sumRgbA / sumA, sumA > 0.0);

  // C) COVERAGE-WEIGHTED PREMULT  (Σ(rgb·a)·0.125). Same as A at mip0→1 but double-counts
  //    coverage deeper → far field over-darkens: DIMMEST, strongest leak suppression.
  // let outRgb = sumRgbA * 0.125;
  textureStore(dst, coord, vec4<f32>(outRgb, outA));
}
`,
);
