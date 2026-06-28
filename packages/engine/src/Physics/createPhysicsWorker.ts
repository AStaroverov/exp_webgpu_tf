// Worker-construction seam (plan §6.4). Returns the live physics Worker and posts the
// SAB bundle as the init message. Steady state crosses ZERO messages after this — pose
// flows through the shared SAB; only structural ops use postMessage.
//
// BROWSER path only for now. A future node:worker_threads training host plugs in here
// (same physics.worker.ts logic, only the host that constructs the Worker + posts
// `init` differs) — left as a clean seam, NOT built (plan §6.4).

import type { SabBundle } from "../../../common/src/sab/registry.ts";

export type PhysicsWorker = {
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

  // Surface worker failures LOUDLY. The pose flows through the SAB with no per-frame
  // messages, so a worker that dies after init is otherwise invisible (the renderer just
  // reads a never-published bank → every body sits at the origin). These handlers turn
  // that silent failure into a console error.
  worker.onerror = (e) => {
    console.error("[physics.worker] uncaught error:", e.message, e.filename, e.lineno, e);
  };
  worker.onmessageerror = (e) => {
    console.error("[physics.worker] message deserialization error:", e);
  };
  worker.onmessage = (ev: MessageEvent) => {
    const msg = ev.data as { type?: string; message?: string; stack?: string };
    if (msg?.type === "ready") {
      console.info("[physics.worker] ready — Rapier initialized, self-clock started");
    } else if (msg?.type === "error") {
      console.error("[physics.worker] reported error:", msg.message, msg.stack);
    }
  };

  // INIT exactly once. Ops posted before the worker replies {type:"ready"} are buffered
  // worker-side and drained at its first phase boundary, so gating on ready is optional.
  console.log(
    "[main] created worker; posting init. dataSab",
    bundle.dataSab.byteLength,
    "controlSab",
    bundle.controlSab.byteLength,
    "isSAB",
    bundle.dataSab instanceof SharedArrayBuffer,
  );
  worker.postMessage({ type: "init", bundle });

  return {
    terminate() {
      worker.terminate();
    },
  };
}
