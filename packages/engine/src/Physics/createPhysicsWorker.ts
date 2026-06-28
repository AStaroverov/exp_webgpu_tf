// Worker-construction seam (plan §6.4). Returns the live physics Worker and posts the
// SAB bundle as the init message. Steady state crosses ZERO messages after this — pose
// flows through the shared SAB; only structural ops use postMessage.
//
// BROWSER path only for now. A future node:worker_threads training host plugs in here
// (same physics.worker.ts logic, only the host that constructs the Worker + posts
// `init` differs) — left as a clean seam, NOT built (plan §6.4).

import type { SabBundle } from "../../../common/src/sab/registry.ts";

export type PhysicsWorker = {
  // Resolves once the worker has loaded Rapier, built its world, and started its
  // self-clock (it posts {type:"ready"}). Rejects if the worker reports an init error
  // or dies uncaught. Callers MUST await this before treating the engine as live — the
  // pose bank stays unpublished until then, and a failed Rapier load is otherwise a
  // silent dead worker (every body reads the origin forever).
  ready: Promise<void>;
  // Structural ops no longer flow through the worker port — main writes them into the
  // OPS ring SAB (engineSab.pushOps) and the worker drains the ring. This seam only
  // constructs the worker, hands over the bundle, and owns teardown.
  terminate(): void;
};

// Construct the physics worker (browser), hand it the SAB bundle, and return the
// op-posting seam. The bundle's SharedArrayBuffers are shared by reference through
// structured clone — NOT transferred (transferring would detach them on main).
export function createPhysicsWorker(bundle: SabBundle): PhysicsWorker {
  const worker = new Worker(new URL("../physics.worker.ts", import.meta.url), {
    type: "module",
  });

  let resolveReady!: () => void;
  let rejectReady!: (err: Error) => void;
  const ready = new Promise<void>((resolve, reject) => {
    resolveReady = resolve;
    rejectReady = reject;
  });

  // Surface worker failures LOUDLY and settle `ready`. The pose flows through the SAB
  // with no per-frame messages, so a worker that dies after init is otherwise invisible
  // (the renderer just reads a never-published bank → every body sits at the origin).
  worker.onerror = (e) => {
    rejectReady(new Error(`physics worker uncaught error: ${e.message} (${e.filename}:${e.lineno})`));
  };
  worker.onmessageerror = (e) => {
    rejectReady(new Error(`physics worker message deserialization error: ${String(e)}`));
  };
  worker.onmessage = (ev: MessageEvent) => {
    const msg = ev.data as { type?: string; message?: string; stack?: string };
    if (msg?.type === "ready") {
      resolveReady();
    } else if (msg?.type === "error") {
      rejectReady(new Error(`physics worker init failed: ${msg.message}\n${msg.stack ?? ""}`));
    }
  };

  // INIT exactly once. Ops posted before the worker replies {type:"ready"} are buffered
  // worker-side (in the OPS ring SAB) and drained at its first phase boundary, so they
  // are never lost — but callers must still await `ready` so an init failure surfaces.
  worker.postMessage({ type: "init", bundle });

  return {
    ready,
    terminate() {
      worker.terminate();
    },
  };
}
