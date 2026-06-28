// Structural-op protocol between the MAIN (render) thread and the PHYSICS worker
// (plan §4.3, SAB op-ring). Structural ops (a body appears / disappears) are written
// into the OPS ring SAB by main and drained by the worker — NO postMessage for ops
// (only the one-time `init` handshake uses postMessage to hand over the SAB bundle).
//
// Steady state crosses ZERO messages — pose flows through the DATA SAB, ops through the
// OPS SAB. This module defines the op TYPES + the fixed-stride record CODEC the ring
// uses (encodeOp on main, decodeOp on the worker). The ring mechanics (cursors, Atomics
// gate) live in engineSab; the SAB layout lives in the ecs package sab/registry.
//
// CONTRACT for the main side: pull `eid = sab.nextEid()` (the one shared authority),
// build the render entity at that eid, then push a SPAWN_BODY carrying that SAME eid
// and the body's geometry. The worker adopts the eid and materializes the Rapier
// body. Despawn pushes DESPAWN_BODY{eid} and is fire-and-forget (eids never recycle,
// plan §6.3) — no ack.

import { OPS_PAYLOAD_STRIDE } from "../../../ECS/src/sab/registry.ts";

// ---- Opcodes ----------------------------------------------------------------

export const OpCode = {
  SPAWN_BODY: 1,
  DESPAWN_BODY: 2,
} as const;
export type OpCode = (typeof OpCode)[number & keyof typeof OpCode];

// ---- Shape payloads ---------------------------------------------------------

export type Vec3 = { x: number; y: number; z: number };

// The body description an entity supplies when it gains RigidBodyState (the public ECS
// surface). `position` is the body CENTER (Rapier translation); center-origin means the
// render z is the SAME center, so the worker mirrors createBody exactly. `density` is
// optional (defaults to 1 in the collider factory). It carries NO eid: the eid is the entity the component is
// added to (RigidBodyState.addComponent stamps it when emitting the op).
export type BodySpec = {
  readonly bodyType: "dynamic" | "fixed";
  readonly position: Vec3; // body CENTER
  readonly density?: number;
} &
  // `halfExtents` = half the box size per axis (a Vec3, like `position`), matching
  // Rapier's `ColliderDesc.cuboid(hx, hy, hz)` which takes half-extents.
  (
    | { readonly kind: "box"; readonly halfExtents: Vec3 }
    // groundBox is just a fixed thin box; a distinct kind so it can be special-cased later.
    | { readonly kind: "groundBox"; readonly halfExtents: Vec3 }
    | { readonly kind: "sphere"; readonly radius: number }
  );

// The wire/ring form: a BodySpec stamped with its eid + the SPAWN opcode.
export type SpawnBodyOp = {
  readonly op: typeof OpCode.SPAWN_BODY;
  readonly eid: number;
} & BodySpec;

export type DespawnBodyOp = {
  readonly op: typeof OpCode.DESPAWN_BODY;
  readonly eid: number;
};

export type StructuralOp = SpawnBodyOp | DespawnBodyOp;

export function toSpawnOp(eid: number, spec: BodySpec): SpawnBodyOp {
  return { op: OpCode.SPAWN_BODY, eid, ...spec };
}

// ---- Lifecycle messages (plan §6.2) -----------------------------------------
// Only the ONE-TIME init handshake uses postMessage now (it hands over the SAB bundle).
// Structural ops travel the OPS ring, not postMessage.

// The SAB bundle is the {dataSab, controlSab, opsSab, layoutVersion} from
// registry.allocate() (structurally cloneable — SharedArrayBuffers are shared, not
// copied). We import it loosely to avoid a hard dep on the registry path here.
export type InitMessage = {
  readonly type: "init";
  readonly bundle: {
    readonly opsSab: SharedArrayBuffer;
    readonly dataSab: SharedArrayBuffer;
    readonly controlSab: SharedArrayBuffer;
    readonly layoutVersion: number;
  };
};

export type ReadyMessage = { readonly type: "ready" };
export type ErrorMessage = { readonly type: "error"; readonly message: string };

// Everything the worker may receive from main (init handshake only).
export type WorkerInbound = InitMessage;
// Everything the worker may post back to main.
export type WorkerOutbound = ReadyMessage | ErrorMessage;

export function isInitMessage(msg: unknown): msg is InitMessage {
  return (
    typeof msg === "object" &&
    msg !== null &&
    (msg as { type?: unknown }).type === "init" &&
    typeof (msg as { bundle?: unknown }).bundle === "object" &&
    (msg as { bundle?: unknown }).bundle !== null
  );
}

export function despawnBody(eid: number): DespawnBodyOp {
  return { op: OpCode.DESPAWN_BODY, eid };
}

// ---- Type narrowing (worker side) -------------------------------------------

export function isSpawnBody(op: StructuralOp): op is SpawnBodyOp {
  return op.op === OpCode.SPAWN_BODY;
}

export function isDespawnBody(op: StructuralOp): op is DespawnBodyOp {
  return op.op === OpCode.DESPAWN_BODY;
}

// ---- OPS-ring record codec (plan §4.3) --------------------------------------
//
// Fixed-stride Float64 record matching OPS_PAYLOAD_STRIDE. The opcode itself rides the
// ring's Int32 STATE slot (the Atomics gate), so the payload holds only the fields:
//   [0] eid  [1] bodyType(0=dynamic,1=fixed)  [2] kind(0=box,1=groundBox,2=sphere)
//   [3..5] position xyz (body CENTER)  [6..8] half-extents xyz (box) | radius in [6] (sphere)
//   [9] density (0 = unset)  [10] reserved
const KIND_CODE = { box: 0, groundBox: 1, sphere: 2 } as const;

// Encode an op into the payload at slot `slot`; returns the opcode for the STATE gate.
export function encodeOp(op: StructuralOp, payload: Float64Array, slot: number): number {
  const b = slot * OPS_PAYLOAD_STRIDE;
  payload[b] = op.eid;
  if (op.op === OpCode.DESPAWN_BODY) return OpCode.DESPAWN_BODY;
  payload[b + 1] = op.bodyType === "fixed" ? 1 : 0;
  payload[b + 2] = KIND_CODE[op.kind];
  payload[b + 3] = op.position.x;
  payload[b + 4] = op.position.y;
  payload[b + 5] = op.position.z;
  if (op.kind === "sphere") {
    payload[b + 6] = op.radius;
  } else {
    payload[b + 6] = op.halfExtents.x;
    payload[b + 7] = op.halfExtents.y;
    payload[b + 8] = op.halfExtents.z;
  }
  payload[b + 9] = op.density ?? 0;
  return OpCode.SPAWN_BODY;
}

// Decode the record at slot `slot` (given the opcode read from the STATE gate).
export function decodeOp(opcode: number, payload: Float64Array, slot: number): StructuralOp {
  const b = slot * OPS_PAYLOAD_STRIDE;
  const eid = payload[b];
  if (opcode === OpCode.DESPAWN_BODY) return { op: OpCode.DESPAWN_BODY, eid };
  const bodyType = payload[b + 1] === 1 ? "fixed" : "dynamic";
  const position: Vec3 = { x: payload[b + 3], y: payload[b + 4], z: payload[b + 5] };
  const density = payload[b + 9] || undefined;
  const kindCode = payload[b + 2];
  if (kindCode === KIND_CODE.sphere) {
    return {
      op: OpCode.SPAWN_BODY,
      eid,
      bodyType,
      kind: "sphere",
      position,
      radius: payload[b + 6],
      density,
    };
  }
  const kind = kindCode === KIND_CODE.groundBox ? "groundBox" : "box";
  return {
    op: OpCode.SPAWN_BODY,
    eid,
    bodyType,
    kind,
    position,
    halfExtents: { x: payload[b + 6], y: payload[b + 7], z: payload[b + 8] },
    density,
  };
}
