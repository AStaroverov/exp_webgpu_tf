import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";
import { SURFEL_CAP } from "./surfelResources.ts";

// Surfel Radiance Cascades — Stage B INSERT compute pass.
//
// One thread per surfel slot (workgroup 64, dispatch ceil(CAP/64)). Each LIVE
// surfel (posr.w > 0) inserts its id into every spatial-hash bucket its
// radius-disc overlaps, building the grid from the CURRENT live set (= last
// frame's survivors). The grid is consumed the same frame by the (later) spawn
// coverage test and recycle over-coverage test.
//
// DEVIATION FROM src-dgi: src-dgi builds the accel grid with count → prefix-sum
// → accelerate (a fragile segmented GPU scan). We use a FIXED-CAPACITY-PER-CELL
// hash grid instead (no prefix-sum: one insert pass + a per-frame clear),
// justified by our FIXED world surfel radius (orthographic camera ⇒ uniform
// cells). A bucket past CELL_K ids DROPS the overflow — acceptable, since
// coverage queries only need a few nearby surfels and recycle bounds density.
//
// GRID LAYOUT (surfelResources.ts): GRID_CAP buckets, stride 1 + CELL_K.
// Bucket b at offset b*(1+CELL_K): [0] = count (atomic), [1..CELL_K] = ids.
//
// surfel_posr / surfel_grid are STANDALONE GPUBuffers (see surfelResources.ts) —
// the VariableMetas below exist ONLY for WGSL emission + the (kind-based,
// size-agnostic) bind-group layout entry. Their atomic / sized-array types are
// never parsed for size. Bind them manually against getBindGroupLayout(2).

// COMPUTE-visibility group-0 uniform.
const uC = (name: string, type: string) =>
  new VariableMeta(name, VariableKind.Uniform, type, {
    group: 0,
    visibility: GPUShaderStage.COMPUTE,
  });

export const shaderMeta = new ShaderMeta(
  {
    // ---- group 0 : per-frame uniforms (COMPUTE-only) ----
    // .x = cellSize (world units), .y = surfelCap, .z = gridCap, .w = cellK.
    params: uC("uParams", `vec4<f32>`),

    // ---- group 2 : surfel storage (StorageWrite => @group(2)) ----
    // xyz = world position, w = radius² (w == 0 => DEAD slot). Read-only here.
    surfelPosr: new VariableMeta(
      "surfel_posr",
      VariableKind.StorageWrite,
      `array<vec4<f32>, ${SURFEL_CAP}>`,
      { group: 2, binding: 0, visibility: GPUShaderStage.COMPUTE },
    ),
    // Spatial hash grid: per bucket [count, id0..id(CELL_K-1)]. atomic<u32> so
    // the whole array stays atomic-typed (atomicAdd on count, atomicStore on ids).
    // Length GRID_CAP*(1+CELL_K) = 65536*17 = 1114112 (see surfelResources.ts).
    surfelGrid: new VariableMeta(
      "surfel_grid",
      VariableKind.StorageWrite,
      `array<atomic<u32>, 1114112>`,
      { group: 2, binding: 1, visibility: GPUShaderStage.COMPUTE },
    ),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
// Solid integer spatial hash (Teschner et al. primes). bitcast i32->u32 so
// negative cell coords hash correctly.
fn hash3(c: vec3<i32>) -> u32 {
  return (bitcast<u32>(c.x) * 73856093u)
       ^ (bitcast<u32>(c.y) * 19349663u)
       ^ (bitcast<u32>(c.z) * 83492791u);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;
  let cap = u32(uParams.y);
  // 1. Bounds check (dispatch is ceil-rounded; drop the over-edge threads).
  if (id >= cap) {
    return;
  }

  // 2. Skip dead slots (posr.w == 0). Live surfels have w = radius² > 0.
  let posr = surfel_posr[id];
  if (posr.w <= 0.0) {
    return;
  }

  let pos = posr.xyz;
  let radius = sqrt(posr.w);
  let cellSize = uParams.x;
  let gridCap = u32(uParams.z);
  let cellK = u32(uParams.w);
  let stride = 1u + cellK;

  // 3. Cell range covering the radius-disc: [floor((p-r)/cs), floor((p+r)/cs)]
  //    per axis. With cellSize >= 2r that is 1-2 cells/axis => up to 8 buckets.
  let lo = vec3<i32>(floor((pos - vec3<f32>(radius)) / cellSize));
  let hi = vec3<i32>(floor((pos + vec3<f32>(radius)) / cellSize));

  // 4. Insert this id into every overlapped bucket.
  for (var cz = lo.z; cz <= hi.z; cz = cz + 1) {
    for (var cy = lo.y; cy <= hi.y; cy = cy + 1) {
      for (var cx = lo.x; cx <= hi.x; cx = cx + 1) {
        let bucket = hash3(vec3<i32>(cx, cy, cz)) % gridCap;
        let base = bucket * stride;
        let slot = atomicAdd(&surfel_grid[base], 1u);
        if (slot < cellK) {
          atomicStore(&surfel_grid[base + 1u + slot], id);
        }
      }
    }
  }
}
    `,
);
