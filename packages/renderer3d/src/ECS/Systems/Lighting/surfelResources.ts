// Surfel Radiance Cascades — Stage A resources.
//
// The three core surfel buffers are STANDALONE GPUBuffers (NOT GPUVariable),
// because their WGSL types use `atomic<u32>` / sized arrays that the JS
// type-size parser cannot understand. Byte sizes are computed here in JS.
// (Matching VariableMetas are declared in the spawn / debug-draw shaders only
// for WGSL emission + bind-group layout — those entries are kind-based and
// size-agnostic, so the parser is never hit there.)

export const SURFEL_CAP = 1 << 16; // 65536

// surfel_stack: u32[1 + CAP]. [0] = allocated count (atomic). [1+i] = free-id pool.
const STACK_LEN = 1 + SURFEL_CAP; // 65537 u32
const STACK_BYTES = STACK_LEN * 4; // (1 + CAP) * 4 = 262148

// surfel_posr: vec4<f32>[CAP] — xyz position, w = radius² (w==0 ⇒ DEAD slot).
const POSR_BYTES = SURFEL_CAP * 16; // CAP * 16 = 1048576

// surfel_norw: vec4<f32>[CAP] — xyz normal, w = recycle marker.
const NORW_BYTES = SURFEL_CAP * 16; // CAP * 16 = 1048576

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
  }

  // WebGPU zero-initializes new buffers, so posr/norw start all-dead. Only the
  // stack needs the identity-pool init.
  initStack();

  return {
    stack,
    posr,
    norw,
    cap: SURFEL_CAP,
    initStack,
    clear,
    destroy(): void {
      stack.destroy();
      posr.destroy();
      norw.destroy();
    },
  };
}
