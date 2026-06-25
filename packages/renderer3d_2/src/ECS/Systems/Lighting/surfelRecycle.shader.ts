import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";
import { SURFEL_CAP } from "./surfelResources.ts";

// Surfel Radiance Cascades — Stage B RECYCLE compute pass.
//
// One thread per surfel slot (workgroup 64, dispatch ceil(CAP/64)). Each LIVE
// surfel (posr.w > 0) decrements its recycle marker (norw.w -= markerDecay) and
// evaluates a recycle heuristic (mirror of src-dgi recycle.slang, simplified):
//
//   heuristic = 0
//   if the (decremented) marker <= 0          => STALE       (heuristic = 1.0)
//   else for each OTHER live surfel sharing this surfel's bucket, if its
//        point_coverage of THIS surfel's center is "over-covered"
//        (nor_dist < recycleCoverage)         => heuristic += 0.2
//
// Then a frame-seeded wang(frameIndex + id) random in [0,1) < heuristic recycles
// the surfel: mark it dead (posr.w = 0) and PUSH its id back onto the free stack
// (mirror src-dgi stack_push): slot = atomicSub(&stack[0],1u) - 1u;
// atomicStore(&stack[1+slot], id). Guard against stack[0] == 0 underflow.
//
// Together with the coverage-gated spawn, this converges the live surfel count to
// ~(visible surface area / (π r²)) and tracks the view as the camera orbits.
//
// STACK SAFETY: spawn POPS the stack (atomicAdd in the spawn pass) and recycle
// PUSHES it (atomicSub here) in SEPARATE encoder passes, so the two never
// interleave on stack[0] within a single pass.
//
// The surfel_stack/posr/norw/grid buffers are STANDALONE GPUBuffers (see
// surfelResources.ts). The VariableMetas below exist ONLY for WGSL emission +
// the (kind-based, size-agnostic) bind-group layout. Their atomic / sized-array
// types are never parsed for size. Bind them manually against
// recyclePipeline.getBindGroupLayout(2).

// COMPUTE-visibility group-0 uniform.
const uC = (name: string, type: string) =>
  new VariableMeta(name, VariableKind.Uniform, type, {
    group: 0,
    visibility: GPUShaderStage.COMPUTE,
  });

export const shaderMeta = new ShaderMeta(
  {
    // ---- group 0 : per-frame uniforms (COMPUTE-only) ----
    // .x = cellSize (world units), .y = gridCap, .z = cellK, .w = frameIndex.
    params: uC("uParams", `vec4<f32>`),
    // .x = surfelCap, .y = markerDecay, .z = recycleCoverage, .w = (unused).
    params2: uC("uParams2", `vec4<f32>`),

    // ---- group 2 : surfel storage (StorageWrite => @group(2)) ----
    // surfel_stack: [0] = atomic allocated count, [1+i] = free-id pool. We PUSH
    // back here on recycle (atomicSub on [0] + atomicStore the freed id).
    surfelStack: new VariableMeta(
      "surfel_stack",
      VariableKind.StorageWrite,
      `array<atomic<u32>, ${SURFEL_CAP + 1}>`,
      { group: 2, binding: 0, visibility: GPUShaderStage.COMPUTE },
    ),
    // xyz = world position, w = radius² (w == 0 => DEAD slot). We set w = 0 to
    // kill a recycled surfel.
    surfelPosr: new VariableMeta(
      "surfel_posr",
      VariableKind.StorageWrite,
      `array<vec4<f32>, ${SURFEL_CAP}>`,
      { group: 2, binding: 1, visibility: GPUShaderStage.COMPUTE },
    ),
    // xyz = world normal, w = recycle marker. We decrement w each frame.
    surfelNorw: new VariableMeta(
      "surfel_norw",
      VariableKind.StorageWrite,
      `array<vec4<f32>, ${SURFEL_CAP}>`,
      { group: 2, binding: 2, visibility: GPUShaderStage.COMPUTE },
    ),
    // Spatial hash grid (built by the INSERT pass; surfelResources.ts). Recycle only
    // READS it (atomicLoad) for the over-coverage test, but WGSL forbids atomic<u32>
    // in a `read` storage var — atomics MUST be read_write. So StorageWrite
    // (read_write); reads are allowed. Lands in @group(2) (binding 3) with the others.
    // Length GRID_CAP*(1+CELL_K) = 65536*17 = 1114112.
    surfelGrid: new VariableMeta(
      "surfel_grid",
      VariableKind.StorageWrite,
      `array<atomic<u32>, 1114112>`,
      { group: 2, binding: 3, visibility: GPUShaderStage.COMPUTE },
    ),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
// Solid integer spatial hash (Teschner et al. primes). bitcast i32->u32 so
// negative cell coords hash correctly. Must match surfelInsert.shader.ts.
fn hash3(c: vec3<i32>) -> u32 {
  return (bitcast<u32>(c.x) * 73856093u)
       ^ (bitcast<u32>(c.y) * 19349663u)
       ^ (bitcast<u32>(c.z) * 83492791u);
}

// Self-contained wang hash / random (matches surfelSpawn.shader.ts).
fn wang_hash(seed_in: u32) -> u32 {
  var seed = seed_in;
  seed = (seed ^ 61u) ^ (seed >> 16u);
  seed = seed * 9u;
  seed = seed ^ (seed >> 4u);
  seed = seed * 0x27d4eb2du;
  seed = seed ^ (seed >> 15u);
  return seed;
}
fn random01(seed: u32) -> f32 {
  return f32(wang_hash(seed)) / 4294967295.0;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;
  let cap = u32(uParams2.x);
  // 1. Bounds check (dispatch is ceil-rounded; drop the over-edge threads).
  if (id >= cap) {
    return;
  }

  // 2. Skip dead slots (posr.w == 0). Live surfels have w = radius² > 0.
  let posr = surfel_posr[id];
  if (posr.w <= 0.0) {
    return;
  }

  let markerDecay = uParams2.y;

  // 3. Decrement the recycle marker (norw.w). Spawn sets it to 2.0; it ages each
  //    frame and eventually goes stale.
  let oldMarker = surfel_norw[id].w;
  let newMarker = oldMarker - markerDecay;
  surfel_norw[id].w = newMarker;

  // 4. Recycle heuristic (src-dgi recycle.slang, simplified).
  var heuristic = 0.0;
  if (newMarker <= 0.0) {
    // Stale: aged out (no spawn re-confirmed coverage here ⇒ marker decayed away).
    heuristic = 1.0;
  } else {
    // Over-coverage: count OTHER live surfels in this surfel's own bucket whose
    // point_coverage of THIS surfel's center is "over-covered" (nor_dist below
    // recycleCoverage ⇒ too close / redundant). Each adds 0.2 to the heuristic.
    let cellSize = uParams.x;
    let gridCap = u32(uParams.y);
    let cellK = u32(uParams.z);
    let recycleCoverage = uParams2.z;
    let stride = 1u + cellK;

    let cell = vec3<i32>(floor(posr.xyz / cellSize));
    let bucket = hash3(cell) % gridCap;
    let base = bucket * stride;
    let cnt = min(atomicLoad(&surfel_grid[base]), cellK);

    for (var k = 0u; k < cnt; k = k + 1u) {
      let nid = atomicLoad(&surfel_grid[base + 1u + k]);
      if (nid == id) {
        continue; // skip self
      }
      let np = surfel_posr[nid];
      if (np.w <= 0.0) {
        continue; // skip dead neighbor
      }
      // point_coverage: nor_dist = distance2(neighbor, thisCenter)/neighbor.w - 1
      // (neighbor.w == radius²). < 0 means thisCenter is inside the neighbor's disc.
      let d = np.xyz - posr.xyz;
      let nd = dot(d, d) / np.w - 1.0;
      if (nd < recycleCoverage) {
        heuristic = heuristic + 0.2;
      }
    }
  }

  // 5. Frame-seeded random recycle. Probability scales with the heuristic.
  let seed = wang_hash(u32(uParams.w) + id);
  if (random01(seed) < heuristic) {
    // Recycle: mark dead, then PUSH the freed id back onto the stack
    // (mirror src-dgi stack_push). Guard underflow: stack[0] should never be 0
    // here (a live surfel implies an earlier pop), but skip the push if it is.
    let prevCount = atomicSub(&surfel_stack[0], 1u);
    if (prevCount == 0u) {
      // Underflow guard: restore and bail without freeing (should not happen).
      atomicAdd(&surfel_stack[0], 1u);
      return;
    }
    surfel_posr[id].w = 0.0;
    let slot = prevCount - 1u;
    atomicStore(&surfel_stack[1u + slot], id);
  }
}
    `,
);
