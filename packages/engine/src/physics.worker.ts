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
import { adoptEntity } from "../../common/src/sab/adoptEntity.ts";
import {
  decodeOp,
  isInitMessage,
  isMoveBody,
  isSpawnBody,
  type MoveBodyOp,
  type SpawnBodyOp,
  type WorkerInbound,
  type WorkerOutbound,
} from "./Physics/opChannel.ts";

// Rapier-touching modules are imported DYNAMICALLY (in onInit) on purpose: the Rapier
// WASM ESM import carries a top-level await (vite-plugin-wasm), and a STATIC import of
// it in a module worker blocks the worker's module body from ever running (onmessage
// never gets registered → the worker looks dead). A dynamic import is async and keeps
// the module body synchronous, so the worker boots immediately and we can try/catch the
// Rapier load.
let initPhysicalWorld: typeof import("./Physics/initPhysicalWorld.ts").initPhysicalWorld | null =
  null;
let spawnBodyFromOp: typeof import("./Physics/spawnBodyFromOp.ts").spawnBodyFromOp | null = null;

// ---- DIAGNOSTIC LOGGING (temporary; remove once the worker is confirmed) -----
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
  (self as unknown as Worker).postMessage(msg);
}

self.onmessage = (ev: MessageEvent<WorkerInbound>) => {
  if (isInitMessage(ev.data)) {
    void onInit(ev.data.bundle);
    return;
  }
};

async function onInit(bundle: {
  opsSab: SharedArrayBuffer;
  dataSab: SharedArrayBuffer;
  controlSab: SharedArrayBuffer;
  layoutVersion: number;
}): Promise<void> {
  try {
    const [physMod, spawnMod] = await Promise.all([
      import("./Physics/initPhysicalWorld.ts"),
      import("./Physics/spawnBodyFromOp.ts"),
    ]);
    initPhysicalWorld = physMod.initPhysicalWorld;
    spawnBodyFromOp = spawnMod.spawnBodyFromOp;

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
  lastWakeMs = performance.now();
  scheduleStep();
}

// ---- Structural ops (drained at the phase boundary, before step) ------------

function drainOps(): void {
  // Drain the OPS ring in cursor order: spawns adopt eids + build bodies, despawns tear
  // down. The worker's engineSab owns the read cursor and zeroes each consumed slot.
  getEngineSab(world!).drainOps((opcode, payload, slot) => {
    const op = decodeOp(opcode, payload, slot);
    if (isSpawnBody(op)) spawnBody(op);
    else if (isMoveBody(op)) moveBody(op);
    else despawnBody(op.eid);
  });
}

function moveBody(op: MoveBodyOp): void {
  const w = world!;
  const pw = physicalWorld!;
  const { RigidBodyRef } = getEngineComponents(w);

  const pid = RigidBodyRef.id[op.eid];
  if (pid !== 0) {
    const body = pw.getRigidBody(pid);
    if (body) {
      body.setTranslation({ x: op.x, y: op.y, z: op.z }, true);
      body.setRotation({ x: op.qx, y: op.qy, z: op.qz, w: op.qw }, true);
    }
  }
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
  // The drained op IS a BodySpec (+ op/eid); pass it as the required spec. Consumer side,
  // so addComponent re-seeds the shared banks (same values) and does NOT re-emit an op.
  RigidBodyState.addComponent(w, op.eid, op);
  pidToEid.set(pid, op.eid);
  bodyCount++;
  console.log(
    "[worker] spawnBody eid",
    op.eid,
    op.kind,
    op.bodyType,
    "-> pid",
    pid,
    "bodies",
    bodyCount,
  );
}

function despawnBody(eid: number): void {
  const w = world!;
  const pw = physicalWorld!;
  const { RigidBodyRef } = getEngineComponents(w);

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

function stepOnce(): void {
  const pw = physicalWorld!;
  const sab = getEngineSab(world!);

  drainOps();
  pw.step();
  syncRigidBodyState!();
  sab.publish((elapsedMs += FIXED_STEP_MS));
}

function tick(): void {
  const t = performance.now();
  accumulatorMs += t - lastWakeMs;
  lastWakeMs = t;

  let substeps = 0;
  while (accumulatorMs >= FIXED_STEP_MS && substeps < MAX_SUBSTEPS) {
    stepOnce();
    accumulatorMs -= FIXED_STEP_MS;
    substeps++;
  }
  if (substeps >= MAX_SUBSTEPS) accumulatorMs = 0;

  scheduleStep();
}

function scheduleStep(): void {
  setTimeout(tick, FIXED_STEP_MS);
}
