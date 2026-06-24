import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";
import { GRID_CAP, SURFEL_CAP } from "./surfelResources.ts";

// Surfel Radiance Cascades — Stage B SPAWN compute pass (coverage-gated).
//
// One thread per G-buffer pixel (workgroup 8x8, dispatch ceil(W/8) x ceil(H/8)).
// Each thread reconstructs the pixel's world position from the reverse-Z depth +
// inverse view-projection (copied verbatim from worldComposite.shader.ts), reads
// the G-buffer normal, then runs a COVERAGE TEST against the spatial-hash grid
// (built by the INSERT pass from last frame's live surfels): if some nearby surfel
// already covers this point it bails (no alloc). Otherwise, gated by a low per-pixel
// random chance, it atomically allocates a surfel slot and writes position(+radius²)
// and normal. Coverage gating + the recycle pass make the live count settle far
// below CAP (Stage A pegged at CAP). No gather/merge/composite yet (stages C/D).
//
// COORDINATES: world is Z-up, ORTHOGRAPHIC tilted camera, reverse-Z (NEAR=1,
// FAR=0). depth <= 0 => far/background => no surface. Surfel radius is a FIXED
// world-unit value (no perspective fov term), squared into posr.w.
//
// The three surfel buffers (surfel_stack/posr/norw) are STANDALONE GPUBuffers
// (see surfelResources.ts) — declared here as StorageWrite VariableMetas ONLY for
// WGSL emission + the (kind-based, size-agnostic) bind-group layout entry. Their
// `atomic<u32>` / sized-array types are never parsed for size, so the JS type-size
// parser is never invoked for them. Bind them manually against
// pipeline.getBindGroupLayout(2).

// COMPUTE-visibility group-0 binding (uniforms + textures must advertise the
// compute stage, else the dispatch fails layout validation).
const uC = (name: string, type: string) =>
  new VariableMeta(name, VariableKind.Uniform, type, {
    group: 0,
    visibility: GPUShaderStage.COMPUTE,
  });

export const shaderMeta = new ShaderMeta(
  {
    // ---- group 0 : per-frame uniforms (COMPUTE-only) ----
    // Inverse view-projection (reverse-Z) for world reconstruction; matches
    // ResizeSystem.viewProjMatrix inverse, exactly like worldComposite.
    invViewProj: uC("uInvViewProj", `mat4x4<f32>`),
    // Packed scalars A: .x = resW, .y = resH, .z = frameIndex (as f32), .w = spawnChance.
    packA: uC("uPackA", `vec4<f32>`),
    // Packed scalars B: .x = surfelRadius (world units), .y = surfelCap (as f32),
    // .z = cellSize (world units), .w = coverageThreshold.
    packB: uC("uPackB", `vec4<f32>`),
    // Packed scalars C: .x = gridCap (as f32), .y = cellK (as f32), .z/.w unused.
    // (gridCap/cellK can't fit in packB's 2 free slots once cellSize/threshold take
    // them; one extra vec4 keeps group-0 uniforms at 6, well under 12.)
    packC: uC("uPackC", `vec4<f32>`),

    // ---- group 0 : G-buffer textures (COMPUTE-only) ----
    // Reverse-Z depth (depth32float). sampleType MUST be "depth" — depth32float is
    // not filterable, so the layout entry has to advertise Depth.
    depthTexture: new VariableMeta("depthTexture", VariableKind.Texture, `texture_depth_2d`, {
      group: 0,
      visibility: GPUShaderStage.COMPUTE,
      textureSampleType: "depth",
    }),
    // G-buffer world normals (packed *0.5+0.5; a = surface mask).
    normalTexture: new VariableMeta("normalTexture", VariableKind.Texture, `texture_2d<f32>`, {
      group: 0,
      visibility: GPUShaderStage.COMPUTE,
    }),

    // ---- group 2 : surfel storage (StorageWrite => @group(2)) ----
    // STANDALONE buffers (surfelResources.ts) — these metas are layout/WGSL-emission
    // only. surfel_stack: [0] = atomic allocated count, [1+i] = free-id pool.
    surfelStack: new VariableMeta(
      "surfel_stack",
      VariableKind.StorageWrite,
      `array<atomic<u32>, ${SURFEL_CAP + 1}>`,
      { group: 2, binding: 0, visibility: GPUShaderStage.COMPUTE },
    ),
    // xyz = world position, w = radius² (w == 0 => DEAD slot).
    surfelPosr: new VariableMeta(
      "surfel_posr",
      VariableKind.StorageWrite,
      `array<vec4<f32>, ${SURFEL_CAP}>`,
      { group: 2, binding: 1, visibility: GPUShaderStage.COMPUTE },
    ),
    // xyz = world normal, w = recycle marker (set 2.0 on spawn).
    surfelNorw: new VariableMeta(
      "surfel_norw",
      VariableKind.StorageWrite,
      `array<vec4<f32>, ${SURFEL_CAP}>`,
      { group: 2, binding: 2, visibility: GPUShaderStage.COMPUTE },
    ),
    // Spatial hash grid (built by the INSERT pass; surfelResources.ts). Spawn only
    // READS it (atomicLoad) for the coverage test, but WGSL forbids atomic<u32> in a
    // `read` storage var — atomics MUST be read_write. So it is StorageWrite
    // (read_write); reading from read_write is allowed. This also lands it in @group(2)
    // alongside stack/posr/norw (binding 3), matching the system's group-2 bind group.
    // Length GRID_CAP*(1+CELL_K) = 65536*17 = 1114112.
    surfelGrid: new VariableMeta(
      "surfel_grid",
      VariableKind.StorageWrite,
      `array<atomic<u32>, 1114112>`,
      { group: 2, binding: 3, visibility: GPUShaderStage.COMPUTE },
    ),
    // Per-bucket "claimed this frame" atomic (surfelResources.ts, cleared each frame).
    // atomicAdd before alloc; only the first claimant of a bucket spawns ⇒ ≤1 new
    // surfel per bucket per frame, immune to spawn chance over-driving the population.
    surfelClaim: new VariableMeta(
      "surfel_claim",
      VariableKind.StorageWrite,
      `array<atomic<u32>, ${GRID_CAP}>`,
      { group: 2, binding: 4, visibility: GPUShaderStage.COMPUTE },
    ),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
// ===== hash / random (self-contained wang/PCG; spawn gate) =====
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

// Solid integer spatial hash — IDENTICAL to surfelInsert.shader.ts (Teschner et al.
// primes; bitcast i32->u32 so negative cell coords hash the same as on insert).
fn hash3(c: vec3<i32>) -> u32 {
  return (bitcast<u32>(c.x) * 73856093u)
       ^ (bitcast<u32>(c.y) * 19349663u)
       ^ (bitcast<u32>(c.z) * 83492791u);
}

// One spawn CANDIDATE per 8x8 workgroup (src-dgi gs_candidate). Without this, every
// pixel races the global atomic stack independently; GPUs run workgroups in ~raster
// order (low index = left/top first), so when free slots are scarce the LEFT side
// always wins the alloc → screen-space-left clusters regardless of scene rotation.
// Picking one (random) candidate per group bounds the flood and spreads spawns.
// Packed: (random24 << 8) | (lid + 1). lid in [0,63] so (lid+1) != 0 marks "valid";
// atomicMax over the 24 random bits picks a RANDOM wanting lane (no fixed-lane bias).
var<workgroup> gs_candidate: atomic<u32>;

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  // Reset the per-workgroup candidate slot (single writer + barrier). MUST be in
  // uniform control flow — so NO early returns before the second barrier below.
  if (lid == 0u) {
    atomicStore(&gs_candidate, 0u);
  }
  workgroupBarrier();

  let px = gid.xy;
  let resW = u32(uPackA.x);
  let resH = u32(uPackA.y);
  let res = vec2<f32>(uPackA.x, uPackA.y);
  let frameIndex = u32(uPackA.z);
  let cellSize = uPackB.z;
  let gridCap = u32(uPackC.x);
  let cellK = u32(uPackC.y);
  let stride = 1u + cellK;

  // Decide candidacy WITHOUT returning (all lanes must reach the barrier). A lane
  // "wants" to spawn iff: in-bounds, has a surface, and is NOT already covered.
  var wants = px.x < resW && px.y < resH;

  let depth = textureLoad(depthTexture, vec2<i32>(px), 0);
  if (depth <= 0.0) { wants = false; } // reverse-Z: 0 == far == background

  let n = textureLoad(normalTexture, vec2<i32>(px), 0);
  if (n.a < 0.5) { wants = false; } // G-buffer surface mask

  // World pos (reverse-Z + inverse VP; verbatim worldComposite) + unpacked normal.
  let uv = (vec2<f32>(px) + 0.5) / res;
  let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
  let wp4 = uInvViewProj * ndc;
  let world = wp4.xyz / wp4.w;
  let normal = normalize(n.rgb * 2.0 - 1.0);

  // COVERAGE TEST: if already covered, this lane does NOT want to spawn — but it STILL
  // re-confirms the covering surfel's marker (so visible surfels persist; off-screen
  // ones decay → recycle). Re-marking happens for ALL covered visible lanes, not just
  // the group's spawn winner.
  if (wants) {
    let cell = vec3<i32>(floor(world / cellSize));
    let base = (hash3(cell) % gridCap) * stride;
    let cnt = min(atomicLoad(&surfel_grid[base]), cellK);
    for (var k: u32 = 0u; k < cnt; k = k + 1u) {
      let sid = atomicLoad(&surfel_grid[base + 1u + k]);
      let spos = surfel_posr[sid];
      if (spos.w > 0.0) {
        let dxyz = spos.xyz - world;
        let nor_dist = dot(dxyz, dxyz) / spos.w - 1.0; // src-dgi point_coverage
        if (nor_dist < uPackB.w) {
          surfel_norw[sid].w = 2.0; // re-confirm covering surfel
          wants = false;
          break;
        }
      }
    }
  }

  // Elect ONE random wanting lane as the group's spawn candidate.
  if (wants) {
    let r = wang_hash(px.x ^ (px.y << 16u) ^ (frameIndex * 2654435761u));
    let candidate = ((r >> 8u) << 8u) | (lid + 1u);
    atomicMax(&gs_candidate, candidate);
  }
  workgroupBarrier();

  let winner = atomicLoad(&gs_candidate);
  if (winner == 0u || (winner & 0xFFu) != (lid + 1u)) {
    return; // no candidate in this group, or this lane isn't the chosen one
  }

  // The chosen lane: roll the per-GROUP spawn chance (temporal throttle), then alloc.
  let seed = px.x ^ (px.y << 16u) ^ (frameIndex * 0x9e3779b9u);
  if (random01(seed) > uPackA.w) {
    return;
  }

  // Per-bucket frame claim: only the FIRST lane to claim this world bucket this frame
  // spawns. Caps new surfels at one per bucket per frame so high spawn chance can't
  // over-spawn duplicates into a still-uncovered cell (coverage uses last frame's grid).
  let claimCell = vec3<i32>(floor(world / cellSize));
  let claimBucket = hash3(claimCell) % gridCap;
  if (atomicAdd(&surfel_claim[claimBucket], 1u) != 0u) {
    return;
  }

  let cap = u32(uPackB.y);
  let ptr = atomicAdd(&surfel_stack[0], 1u);
  if (ptr >= cap) {
    // Pool exhausted — CLAMP the count back to cap (src-dgi atomic_set), else the
    // runaway count corrupts the free-list (see the recycle OOB note).
    atomicStore(&surfel_stack[0], cap);
    return;
  }
  let id = atomicLoad(&surfel_stack[1u + ptr]);
  let radius = uPackB.x;
  surfel_posr[id] = vec4<f32>(world, radius * radius);
  surfel_norw[id] = vec4<f32>(normal, 2.0);
}
    `,
);
