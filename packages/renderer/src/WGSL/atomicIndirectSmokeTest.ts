import { VariableKind, VariableMeta } from "../Struct/VariableMeta.ts";
import { ShaderMeta } from "./ShaderMeta.ts";
import { GPUShader } from "./GPUShader.ts";
import { wgsl } from "./wgsl.ts";

// De-risking smoke test for the two GPU primitives the adaptive screen-probe atlas
// needs and which are NOT yet exercised anywhere in this repo:
//   (A) atomics on a storage buffer   — atomicAdd as a lock-free append allocator.
//   (B) compute-written indirect args — a compute pass fills a [G,1,1] args buffer
//       that a SECOND pass consumes via dispatchWorkgroupsIndirect (probe count is
//       known only on the GPU, so the launch size must be produced on-GPU too).
// Runtime PASS/FAIL only, no rendering. Mirrors computeSmokeTest.ts: ShaderMeta with
// StorageWrite (read_write) buffers, autoLayout compute pipelines, COPY_SRC → staging
// read-back, and a single console.log per sub-test.
//
// NOTE on atomic types: getTypeSize/getTypeBufferSize can't parse `atomic<u32>`, so
// those VariableMeta pass explicit size/bufferSize (1 elem / 4 bytes) to bypass the
// type parser — the string still flows verbatim into the generated WGSL declaration.

// ── Sub-test (A) sizing ──────────────────────────────────────────────────────────
const ATOMIC_T = 1024; // threads, each appends exactly once
const ATOMIC_WG = 64; // workgroup size → ATOMIC_T / ATOMIC_WG dispatched groups

// ── Sub-test (B) sizing ──────────────────────────────────────────────────────────
const INDIRECT_WG = 64; // pass-2 workgroup size
const IN_COUNT = 500; // "probe count" the GPU reads to derive the launch size
const EXPECTED_G = Math.ceil(IN_COUNT / INDIRECT_WG); // = 8, what pass 1 must compute
const EXPECTED_LAUNCH = EXPECTED_G * INDIRECT_WG; // = 512, total invocations pass 2 runs

// ── Sub-test (A): atomic append ───────────────────────────────────────────────────
// counter is a bare atomic<u32> slot allocator; data[] receives each thread's gid at
// its atomically-claimed slot. A correct run leaves counter == T and data a permutation
// of {0..T-1} (every thread got a unique slot — no lost/duplicated appends).
const atomicMeta = new ShaderMeta(
  {
    counter: new VariableMeta("uCounter", VariableKind.StorageWrite, `atomic<u32>`, {
      visibility: GPUShaderStage.COMPUTE,
      size: 1,
      bufferSize: 4,
    }),
    data: new VariableMeta("uData", VariableKind.StorageWrite, `array<u32, ${ATOMIC_T}>`, {
      visibility: GPUShaderStage.COMPUTE,
    }),
  },
  {},
  wgsl /* wgsl */ `
    @compute @workgroup_size(${ATOMIC_WG})
    fn append(@builtin(global_invocation_id) gid: vec3<u32>) {
      let slot = atomicAdd(&uCounter, 1u);
      uData[slot] = gid.x;
    }
  `,
);

async function atomicAppendTest(device: GPUDevice): Promise<boolean> {
  const shader = new GPUShader(atomicMeta);
  // Both buffers are StorageWrite → group 2 (bindings 0/1); autoLayout reflects it.
  const pipeline = shader.getComputePipeline(device, "append", { autoLayout: true });
  const counterBuf = shader.uniforms.counter.getGPUBuffer(device);
  const dataBuf = shader.uniforms.data.getGPUBuffer(device);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(2),
    entries: [
      shader.uniforms.counter.getBindGroupEntry(device),
      shader.uniforms.data.getBindGroupEntry(device),
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(2, bindGroup);
  pass.dispatchWorkgroups(ATOMIC_T / ATOMIC_WG);
  pass.end();

  const counterStaging = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const dataStaging = device.createBuffer({
    size: ATOMIC_T * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyBufferToBuffer(counterBuf, 0, counterStaging, 0, 4);
  encoder.copyBufferToBuffer(dataBuf, 0, dataStaging, 0, ATOMIC_T * 4);
  device.queue.submit([encoder.finish()]);

  await Promise.all([
    counterStaging.mapAsync(GPUMapMode.READ),
    dataStaging.mapAsync(GPUMapMode.READ),
  ]);
  const counter = new Uint32Array(counterStaging.getMappedRange().slice(0))[0];
  const data = new Uint32Array(dataStaging.getMappedRange().slice(0));
  counterStaging.unmap();
  dataStaging.unmap();
  counterStaging.destroy();
  dataStaging.destroy();
  shader.destroy();

  // Verdict: counter counted every append, and the appended gids are exactly {0..T-1}
  // (each slot written once). A dropped/duplicated atomic shows up as a hole/collision.
  let ok = counter === ATOMIC_T;
  const seen = new Uint8Array(ATOMIC_T);
  for (let i = 0; i < ATOMIC_T && ok; i++) {
    const v = data[i];
    if (v >= ATOMIC_T || seen[v] === 1) {
      ok = false;
    } else {
      seen[v] = 1;
    }
  }
  console.log(
    `[atomic-indirect-smoke] (A) atomic-append ${ok ? "PASS" : "FAIL"} — counter=${counter} (expected ${ATOMIC_T}), data is${ok ? "" : " NOT"} a permutation of {0..${ATOMIC_T - 1}}`,
  );
  return ok;
}

// ── Sub-test (B): compute-written indirect dispatch ───────────────────────────────
// Pass 1 (1 thread) reads uInCount and writes uArgs = [ceil(count/WG), 1, 1]. uArgs is
// a RAW buffer (STORAGE|INDIRECT|COPY_DST|COPY_SRC) because GPUVariable's STORAGE_USAGE
// has no INDIRECT bit — it's declared in the meta only to generate the WGSL binding, and
// bound manually (the passBuf/mipBuf pattern in createVoxelSystem.ts).
const argsMeta = new ShaderMeta(
  {
    inCount: new VariableMeta("uInCount", VariableKind.StorageRead, `u32`, {
      visibility: GPUShaderStage.COMPUTE,
    }),
    args: new VariableMeta("uArgs", VariableKind.StorageWrite, `array<u32, 3>`, {
      visibility: GPUShaderStage.COMPUTE,
    }),
  },
  {},
  wgsl /* wgsl */ `
    @compute @workgroup_size(1)
    fn writeArgs() {
      let g = (uInCount + ${INDIRECT_WG}u - 1u) / ${INDIRECT_WG}u;
      uArgs[0] = g;
      uArgs[1] = 1u;
      uArgs[2] = 1u;
    }
  `,
);

// Pass 2 is launched by dispatchWorkgroupsIndirect(uArgs). Every invocation bumps a
// single atomic launch counter, so the final value == (groups the GPU actually launched)
// × WG — proving the indirect args consumed were the ones pass 1 wrote.
const launchMeta = new ShaderMeta(
  {
    launched: new VariableMeta("uLaunched", VariableKind.StorageWrite, `atomic<u32>`, {
      visibility: GPUShaderStage.COMPUTE,
      size: 1,
      bufferSize: 4,
    }),
  },
  {},
  wgsl /* wgsl */ `
    @compute @workgroup_size(${INDIRECT_WG})
    fn launch() {
      atomicAdd(&uLaunched, 1u);
    }
  `,
);

async function indirectDispatchTest(device: GPUDevice): Promise<boolean> {
  const argsShader = new GPUShader(argsMeta);
  const launchShader = new GPUShader(launchMeta);

  // uInCount (StorageRead → group 1) seeded with the "probe count".
  const inCountBuf = argsShader.uniforms.inCount.getGPUBuffer(device);
  device.queue.writeBuffer(inCountBuf, 0, new Uint32Array([IN_COUNT]));

  // RAW indirect args buffer — the one bit GPUVariable can't express (INDIRECT).
  const argsBuf = device.createBuffer({
    size: 3 * 4,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.INDIRECT |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });

  const argsPipeline = argsShader.getComputePipeline(device, "writeArgs", { autoLayout: true });
  const launchPipeline = launchShader.getComputePipeline(device, "launch", { autoLayout: true });

  const launchedBuf = launchShader.uniforms.launched.getGPUBuffer(device);

  // Pass 1 bindings: uInCount at its group-1 layout, uArgs bound to the RAW buffer at
  // its group-2 binding (NOT argsShader.uniforms.args.getGPUBuffer — that's the unused
  // GPUVariable buffer; we substitute our own INDIRECT-capable buffer here).
  const argsGroup1 = device.createBindGroup({
    layout: argsPipeline.getBindGroupLayout(1),
    entries: [argsShader.uniforms.inCount.getBindGroupEntry(device)],
  });
  const argsGroup2 = device.createBindGroup({
    layout: argsPipeline.getBindGroupLayout(2),
    entries: [{ binding: argsMeta.uniforms.args.binding, resource: { buffer: argsBuf } }],
  });
  const launchGroup2 = device.createBindGroup({
    layout: launchPipeline.getBindGroupLayout(2),
    entries: [launchShader.uniforms.launched.getBindGroupEntry(device)],
  });

  const encoder = device.createCommandEncoder();
  // Separate compute passes → the implicit inter-pass barrier orders the args write
  // BEFORE the indirect dispatch reads it (same barrier the voxel passes rely on).
  {
    const p1 = encoder.beginComputePass();
    p1.setPipeline(argsPipeline);
    p1.setBindGroup(1, argsGroup1);
    p1.setBindGroup(2, argsGroup2);
    p1.dispatchWorkgroups(1);
    p1.end();
  }
  {
    const p2 = encoder.beginComputePass();
    p2.setPipeline(launchPipeline);
    p2.setBindGroup(2, launchGroup2);
    p2.dispatchWorkgroupsIndirect(argsBuf, 0);
    p2.end();
  }

  const argsStaging = device.createBuffer({
    size: 3 * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const launchedStaging = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyBufferToBuffer(argsBuf, 0, argsStaging, 0, 3 * 4);
  encoder.copyBufferToBuffer(launchedBuf, 0, launchedStaging, 0, 4);
  device.queue.submit([encoder.finish()]);

  await Promise.all([
    argsStaging.mapAsync(GPUMapMode.READ),
    launchedStaging.mapAsync(GPUMapMode.READ),
  ]);
  const args = new Uint32Array(argsStaging.getMappedRange().slice(0));
  const launched = new Uint32Array(launchedStaging.getMappedRange().slice(0))[0];
  argsStaging.unmap();
  launchedStaging.unmap();
  argsStaging.destroy();
  launchedStaging.destroy();
  argsBuf.destroy();
  argsShader.destroy();
  launchShader.destroy();

  // Verdict: the args buffer holds [G,1,1] and pass 2 ran exactly G×WG invocations →
  // the GPU consumed the compute-written args, not a CPU-set fallback.
  const argsOk = args[0] === EXPECTED_G && args[1] === 1 && args[2] === 1;
  const launchOk = launched === EXPECTED_LAUNCH;
  const ok = argsOk && launchOk;
  console.log(
    `[atomic-indirect-smoke] (B) indirect-dispatch ${ok ? "PASS" : "FAIL"} — args=[${args.join(", ")}] (expected [${EXPECTED_G}, 1, 1]), launched=${launched} (expected ${EXPECTED_LAUNCH})`,
  );
  return ok;
}

export async function atomicIndirectSmokeTest(device: GPUDevice): Promise<void> {
  const a = await atomicAppendTest(device);
  const b = await indirectDispatchTest(device);
  const ok = a && b;
  console.log(
    `[atomic-indirect-smoke] ${ok ? "PASS" : "FAIL"} — atomics + compute-written indirect dispatch ${ok ? "verified" : "FAILED"}`,
  );
}
