// PHYSICS WORKER (plan §6.1/§7). Runs Rapier off the main thread; produces pose into
// the shared SAB banks; consumes structural ops (spawn/despawn) over postMessage.
// Steady state exchanges ZERO messages with main — pose flows through the SAB and the
// Atomics SEQ seqcount; only spawn/despawn cross the wire.
//
// Worker construction (BROWSER path) is owned by the main side:
//   new Worker(new URL("./physics.worker.ts", import.meta.url), { type: "module" })
// A future node:worker_threads training path is a clean seam (this file is pure
// message-driven logic; only the host that posts `init` differs) — NOT built here.

import { removeEntity } from "bitecs";
import {
  createPhysicsWorkerWorld,
  getEngineComponents,
  getEngineSab,
  type PhysicsWorkerWorld,
} from "./ECS/createEngineWorld.ts";
import type { PhysicalWorld } from "./Physics/initPhysicalWorld.ts";
import { createRigidBodyStateSystem } from "./ECS/Systems/createRigidBodyStateSystem.ts";
import { adoptEntity } from "../../renderer3d_2/src/sab/adoptEntity.ts";

// Rapier-touching modules are imported DYNAMICALLY (in onInit) on purpose: the Rapier
// WASM ESM import carries a top-level await (vite-plugin-wasm), and a STATIC import of
// it in a module worker blocks the worker's module body from ever running (onmessage
// never gets registered → the worker looks dead). A dynamic import is async and keeps
// the module body synchronous, so the worker boots immediately and we can try/catch the
// Rapier load.
let initPhysicalWorld: typeof import("./Physics/initPhysicalWorld.ts").initPhysicalWorld | null =
  null;
let spawnBodyFromOp: typeof import("./Physics/spawnBodyFromOp.ts").spawnBodyFromOp | null = null;
import {
  isInitMessage,
  isSpawnBody,
  type SpawnBodyOp,
  type WorkerInbound,
  type WorkerOutbound,
} from "./Physics/opChannel.ts";

// ---- DIAGNOSTIC LOGGING (temporary; remove once the worker is confirmed) -----
console.log("[worker] module evaluated (imports + Rapier WASM TLA resolved)");
let stepLog = 0;
let bodyCount = 0;

// ---- Worker-local state -----------------------------------------------------

let world: PhysicsWorkerWorld | null = null;
let physicalWorld: PhysicalWorld | null = null;
let syncRigidBodyState: (() => void) | null = null;

// pid → eid, worker-local (plan §6.1). RigidBodyRef keeps its OWN copy of this map,
// but despawn needs pid→body lookup keyed by eid's stored pid, so we read pid from the
// shared RigidBodyRef.id column directly; this map is kept for symmetry / future
// collision-event resolution and is the worker's authoritative reverse index.
const pidToEid = new Map<number, number>();

// Ops written by main into the OPS ring SAB before the world is READY simply wait there
// (the ring IS the buffer); the self-clock loop drains them at its phase boundary, which
// only starts AFTER onInit builds the world. No separate pending-ops buffer is needed.

// ---- Fixed-step self-clock (plan §7) ----------------------------------------

const FIXED_STEP_MS = 1000 / 60; // matches Rapier's internal fixed step
const MAX_SUBSTEPS = 5; // cap to avoid spiral-of-death after a long stall
let accumulatorMs = 0;
let lastWakeMs = 0;
let elapsedMs = 0;

// ---- Message handling -------------------------------------------------------

function post(msg: WorkerOutbound): void {
  // eslint-disable-next-line no-restricted-globals -- worker global scope
  (self as unknown as Worker).postMessage(msg);
}

self.onmessage = (ev: MessageEvent<WorkerInbound>) => {
  const msg = ev.data;
  console.log("[worker] onmessage:", (msg as { type?: string })?.type);

  if (isInitMessage(msg)) {
    void onInit(msg.bundle);
    return;
  }

  // Structural ops no longer arrive by message — they live in the OPS ring SAB and are
  // drained in the step loop. The only inbound message is `init`.
};

async function onInit(bundle: {
  dataSab: SharedArrayBuffer;
  controlSab: SharedArrayBuffer;
  opsSab: SharedArrayBuffer;
  layoutVersion: number;
}): Promise<void> {
  // Build the physics-only mirror world bound to the RECEIVED SABs (same bytes as
  // main), then the Rapier world (Z-up gravity, reserveHandleZero). After both exist
  // the worker is READY and can materialize bodies.
  //
  // Wrapped: if Rapier WASM fails to init in the worker, or world build throws, report
  // it to main (which logs it) instead of dying silently — a dead worker shows up only
  // as every body frozen at the origin (the never-published bank).
  try {
    console.log("[worker] onInit: dynamically importing Rapier modules…");
    const [physMod, spawnMod] = await Promise.all([
      import("./Physics/initPhysicalWorld.ts"),
      import("./Physics/spawnBodyFromOp.ts"),
    ]);
    initPhysicalWorld = physMod.initPhysicalWorld;
    spawnBodyFromOp = spawnMod.spawnBodyFromOp;
    console.log("[worker] onInit: Rapier modules loaded; building worlds…");

    world = createPhysicsWorkerWorld(bundle);
    physicalWorld = initPhysicalWorld();
    syncRigidBodyState = createRigidBodyStateSystem(world, physicalWorld);
    console.log("[worker] onInit: world + Rapier world built");
  } catch (err) {
    post({ type: "error", message: `init failed: ${(err as Error)?.stack ?? String(err)}` });
    return;
  }

  post({ type: "ready" });

  // Kick the self-clock. From here the worker free-runs independently of main's rAF.
  lastWakeMs = now();
  scheduleStep();
}

// ---- Structural ops (drained at the phase boundary, before step) ------------

function drainOps(): void {
  // Drain the OPS ring in cursor order: spawns adopt eids + build bodies, despawns tear
  // down. The worker's engineSab owns the read cursor and zeroes each consumed slot.
  getEngineSab(world!).drainOps((op) => {
    if (isSpawnBody(op)) spawnBody(op);
    else despawnBody(op.eid);
  });
}

function spawnBody(op: SpawnBodyOp): void {
  const w = world!;
  const pw = physicalWorld!;
  const { RigidBodyRef, RigidBodyState } = getEngineComponents(w);

  // Adopt the EXACT eid main pulled from the shared NEXT_EID counter (plan §4.2): eid
  // N is the same logical entity in both worlds by construction.
  adoptEntity(w, op.eid);

  // Build the Rapier body+collider → pid, then publish pid into the worker's
  // RigidBodyRef (also registers its own pid→eid map internally).
  const pid = spawnBodyFromOp!(pw, op);
  RigidBodyRef.addComponent(w, op.eid, pid);
  RigidBodyState.addComponent(w, op.eid);
  pidToEid.set(pid, op.eid);
  bodyCount++;
  console.log("[worker] spawnBody eid", op.eid, op.kind, op.bodyType, "-> pid", pid, "bodies", bodyCount);
}

function despawnBody(eid: number): void {
  const w = world!;
  const pw = physicalWorld!;
  const { RigidBodyRef } = getEngineComponents(w);

  // Fire-and-forget (plan §6.3): no ack. eids are never recycled, so a late pose write
  // would land in a dead row no query reads — harmless. Remove the Rapier body, drop
  // the reverse map, then remove the entity from the worker world.
  const pid = RigidBodyRef.id[eid];
  if (pid !== 0) {
    const body = pw.getRigidBody(pid);
    if (body) pw.removeRigidBody(body);
    pidToEid.delete(pid);
    RigidBodyRef.clear(eid);
  }
  removeEntity(w, eid);
}

// ---- Self-clocked fixed-step loop (plan §7) ---------------------------------

function now(): number {
  // performance.now exists in worker scope; fall back to Date for node hosts.
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

function stepOnce(): void {
  const pw = physicalWorld!;
  const sab = getEngineSab(world!);

  // Phase boundary: structural changes happen BEFORE step, never mid-query (plan §4.3).
  drainOps();

  pw.step(); // Rapier integrates the fixed step (gravity)
  syncRigidBodyState!(); // Rapier bodies → RigidBodyState back bank (ALL bodies)

  elapsedMs += FIXED_STEP_MS;
  world!.time.elapsed = elapsedMs;
  // Publish the just-written bank: flip SEQ so main reads a complete, tear-free pose.
  sab.publish(elapsedMs);

  if (stepLog < 10) {
    const c = getEngineComponents(world!);
    // sample the FIRST adopted body's freshly-written pose (writeBank was current bank)
    const sampleEid = pidToEid.size > 0 ? [...pidToEid.values()][0] : -1;
    const px = sampleEid >= 0 ? c.RigidBodyState.position.get(sampleEid, 2) : NaN;
    console.log(
      "[worker] step#", stepLog,
      "bodies", bodyCount,
      "SEQ", Atomics.load(sab.control, 1),
      "sampleEid", sampleEid, "z", px,
    );
    stepLog++;
  }
}

function tick(): void {
  const t = now();
  accumulatorMs += t - lastWakeMs;
  lastWakeMs = t;

  let substeps = 0;
  try {
    while (accumulatorMs >= FIXED_STEP_MS && substeps < MAX_SUBSTEPS) {
      stepOnce();
      accumulatorMs -= FIXED_STEP_MS;
      substeps++;
    }
  } catch (err) {
    // Report once and stop the loop — otherwise a per-step throw (e.g. a bad op or a
    // Rapier handle) would spam every frame. Main logs this.
    post({ type: "error", message: `step failed: ${(err as Error)?.stack ?? String(err)}` });
    return; // do not reschedule
  }
  // If we hit the substep cap we're behind real time; drop the backlog so we don't
  // spiral (better to run slow-mo than to freeze catching up).
  if (substeps >= MAX_SUBSTEPS) accumulatorMs = 0;

  scheduleStep();
}

// rAF is unavailable in workers; pace with setTimeout. The accumulator decouples the
// (coarse) timer cadence from the fixed physics step, so jitter in setTimeout does not
// change how many steps run — only when.
function scheduleStep(): void {
  setTimeout(tick, FIXED_STEP_MS);
}
