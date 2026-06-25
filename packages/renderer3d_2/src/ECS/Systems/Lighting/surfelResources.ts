// Surfel Radiance Cascades — Stage A resources.
//
// The three core surfel buffers are STANDALONE GPUBuffers (NOT GPUVariable),
// because their WGSL types use `atomic<u32>` / sized arrays that the JS
// type-size parser cannot understand. Byte sizes are computed here in JS.
// (Matching VariableMetas are declared in the spawn / debug-draw shaders only
// for WGSL emission + bind-group layout — those entries are kind-based and
// size-agnostic, so the parser is never hit there.)

export const SURFEL_CAP = 1 << 16; // 65536

// --- Stage C: per-surfel radiance cache directions ------------------------
// Octahedral direction tile per surfel (cascade 0). DIR0_W=4 ⇒ 16 dirs over the
// FULL sphere; the composite cosine integral selects the receiving hemisphere.
export const SURFEL_DIR0_W = 4;
export const SURFEL_DIR_COUNT = SURFEL_DIR0_W * SURFEL_DIR0_W; // 16

// --- Stage B: spatial hash grid -------------------------------------------
// DEVIATION FROM src-dgi: src-dgi builds the accel grid with count →
// prefix-sum → accelerate (a segmented GPU scan, fragile). We REPLACE that
// with a FIXED-CAPACITY-PER-CELL hash grid (like src-dgi's SASCell
// { count; ids[K] }), which needs NO prefix-sum — just one insert pass + a
// per-frame clear. Justified because our surfels have a FIXED world radius
// (orthographic camera) ⇒ uniform cells. Trade-off: a bucket holding more
// than CELL_K surfels DROPS the overflow — acceptable (coverage queries only
// need a few nearby surfels, and recycle keeps density bounded).
export const GRID_CAP = 65536; // number of hash buckets
export const CELL_K = 16; // max surfel ids stored per bucket

// Bucket layout: [0] = count (atomic), [1..CELL_K] = surfel ids. Stride 1+K.
const GRID_STRIDE = 1 + CELL_K; // 17 u32 per bucket
const GRID_LEN = GRID_CAP * GRID_STRIDE; // 1,114,112 u32 (== WGSL array length)
const GRID_BYTES = GRID_LEN * 4; // 4,456,448 bytes

// surfel_stack: u32[1 + CAP]. [0] = allocated count (atomic). [1+i] = free-id pool.
const STACK_LEN = 1 + SURFEL_CAP; // 65537 u32
const STACK_BYTES = STACK_LEN * 4; // (1 + CAP) * 4 = 262148

// surfel_posr: vec4<f32>[CAP] — xyz position, w = radius² (w==0 ⇒ DEAD slot).
const POSR_BYTES = SURFEL_CAP * 16; // CAP * 16 = 1048576

// surfel_norw: vec4<f32>[CAP] — xyz normal, w = recycle marker.
const NORW_BYTES = SURFEL_CAP * 16; // CAP * 16 = 1048576

// surfel_rad: vec4<f32>[CAP * DIR_COUNT] — per-surfel radiance cache (Stage C).
// rgb = incoming radiance for that octahedral direction interval, a = visibility
// (1 = ray passed unobstructed). Index: rad[surfelId * DIR_COUNT + (v*DIR0_W + u)].
const RAD_LEN = SURFEL_CAP * SURFEL_DIR_COUNT; // 1,048,576 vec4
const RAD_BYTES = RAD_LEN * 16; // 16,777,216 bytes (16 MB)

// STORAGE: compute write + debug-draw read. COPY_DST: init + clear writes.
// COPY_SRC: read-back of stack[0] for diagnostics (kept on all three for symmetry
// with the existing STORAGE_USAGE convention).
const STORAGE_USAGE =
  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

export type SurfelResources = ReturnType<typeof createSurfelResources>;

// Identity free-id pool with count 0: [0 (count), 0, 1, 2, ..., CAP-1].
function makeInitialStack(): Uint32Array {
  const data = new Uint32Array(STACK_LEN);
  data[0] = 0; // allocated count
  for (let i = 0; i < SURFEL_CAP; i++) {
    data[1 + i] = i; // free-id pool identity
  }
  return data;
}

export function createSurfelResources(device: GPUDevice) {
  const stack = device.createBuffer({
    label: "surfel_stack",
    size: STACK_BYTES,
    usage: STORAGE_USAGE,
  });
  const posr = device.createBuffer({
    label: "surfel_posr",
    size: POSR_BYTES,
    usage: STORAGE_USAGE,
  });
  const norw = device.createBuffer({
    label: "surfel_norw",
    size: NORW_BYTES,
    usage: STORAGE_USAGE,
  });

  // surfel_grid: u32[GRID_CAP * (1 + CELL_K)] — the Stage-B hash grid (see the
  // DEVIATION note above). STORAGE: insert writes / coverage reads. COPY_DST:
  // the per-frame clear via encoder.clearBuffer. No COPY_SRC — never read back.
  const grid = device.createBuffer({
    label: "surfel_grid",
    size: GRID_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // surfel_claim: u32[GRID_CAP] — per-bucket "claimed this frame" atomic. Cleared
  // each frame (with the grid). Spawn does atomicAdd before allocating; only the
  // FIRST claimant of a bucket spawns ⇒ at most ONE new surfel per bucket per frame,
  // regardless of spawn chance. Without this, coverage uses LAST frame's grid, so at
  // high spawn chance many screen groups spawn into the same still-empty world cell in
  // one frame (duplicates) → population saturates near cap → free slots scarce → the
  // global stack atomic (raster/left-first order) starves newly-revealed regions.
  const claim = device.createBuffer({
    label: "surfel_claim",
    size: GRID_CAP * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // surfel_rad: vec4<f32>[CAP * DIR_COUNT] — the Stage-C per-surfel radiance cache.
  // STANDALONE (sized array, no atomics ⇒ ordinary read/read_write storage; never a
  // GPUVariable). STORAGE: gather writes / composite reads. COPY_DST: clear() zero.
  // COPY_SRC: kept for diagnostics read-back (STORAGE_USAGE convention).
  const rad = device.createBuffer({
    label: "surfel_rad",
    size: RAD_BYTES,
    usage: STORAGE_USAGE,
  });

  // (Re)initialize the stack to count 0 + identity free-id pool.
  function initStack(): void {
    device.queue.writeBuffer(stack, 0, makeInitialStack());
  }

  // Clear all surfels: reset the stack AND zero posr (w==0 ⇒ DEAD), so the debug
  // draw shows nothing until they respawn. norw is left as-is (w is only a recycle
  // marker; posr.w==0 already kills the slot), matching the prompt's "zero posr".
  function clear(): void {
    initStack();
    device.queue.writeBuffer(posr, 0, new Float32Array(SURFEL_CAP * 4));
    // Zero the radiance cache too, so a cleared scene composites no stale light.
    device.queue.writeBuffer(rad, 0, new Float32Array(RAD_LEN * 4));
  }

  // WebGPU zero-initializes new buffers, so posr/norw start all-dead. Only the
  // stack needs the identity-pool init.
  initStack();

  // Zero the whole hash grid. Takes the frame's command encoder so the clear is
  // ORDERED in the per-frame command stream (right before the insert pass), not
  // a stray device.queue op. The grid is rebuilt from scratch every frame, so
  // clear() (above) does not need to touch it.
  function clearGrid(encoder: GPUCommandEncoder): void {
    encoder.clearBuffer(grid); // whole-buffer clear to 0
    encoder.clearBuffer(claim); // per-bucket frame claim reset
  }

  return {
    stack,
    posr,
    norw,
    grid,
    claim,
    rad,
    cap: SURFEL_CAP,
    initStack,
    clear,
    clearGrid,
    destroy(): void {
      stack.destroy();
      posr.destroy();
      norw.destroy();
      grid.destroy();
      claim.destroy();
      rad.destroy();
    },
  };
}
