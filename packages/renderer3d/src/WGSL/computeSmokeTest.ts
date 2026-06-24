import { VariableKind, VariableMeta } from "../Struct/VariableMeta.ts";
import { ShaderMeta } from "./ShaderMeta.ts";
import { GPUShader } from "./GPUShader.ts";
import { wgsl } from "./wgsl.ts";

// Stage-0 compute-infrastructure smoke test. Exercises the whole new compute path
// end-to-end: a ShaderMeta with a read_write storage buffer (StorageWrite) +
// a @compute entry, getComputePipeline, a dispatch, and a read-back. If this logs
// PASS, the compute foundation (pipeline + storage-write WGSL + COMPUTE-visibility
// bindings + buffer read-back) works and the surfel stages can build on it.

const N = 64;

const smokeMeta = new ShaderMeta(
  {
    // read_write storage (group 2 by kind); COMPUTE visibility so the bind-group
    // layout includes the compute stage.
    out: new VariableMeta("uOut", VariableKind.StorageWrite, `array<u32, ${N}>`, {
      visibility: GPUShaderStage.COMPUTE,
    }),
  },
  {},
  wgsl /* wgsl */ `
    @compute @workgroup_size(${N})
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      if (gid.x < ${N}u) {
        uOut[gid.x] = gid.x * 2u;
      }
    }
  `,
);

export async function computeSmokeTest(device: GPUDevice): Promise<boolean> {
  const shader = new GPUShader(smokeMeta);
  // autoLayout: the storage buffer lands at @group(2) (StorageWrite kind); "auto"
  // reflects that directly, sidestepping the explicit-layout group-index packing.
  const pipeline = shader.getComputePipeline(device, "main", { autoLayout: true });
  const outBuf = shader.uniforms.out.getGPUBuffer(device);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(2),
    entries: [shader.uniforms.out.getBindGroupEntry(device)],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(2, bindGroup);
  pass.dispatchWorkgroups(1);
  pass.end();

  const staging = device.createBuffer({
    size: N * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyBufferToBuffer(outBuf, 0, staging, 0, N * 4);
  device.queue.submit([encoder.finish()]);

  await staging.mapAsync(GPUMapMode.READ);
  const got = new Uint32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  shader.destroy();

  let ok = true;
  for (let i = 0; i < N; i++) {
    if (got[i] !== i * 2) {
      ok = false;
      break;
    }
  }
  console.log(
    `[compute-smoke] ${ok ? "PASS" : "FAIL"} — first 8: [${Array.from(got.slice(0, 8)).join(", ")}]`,
  );
  return ok;
}
