import { OPS_PAYLOAD_STRIDE } from "../../../common/src/sab/registry.ts";

// ---- Opcodes ----------------------------------------------------------------

export const OpCode = {
  SPAWN_BODY: 1,
  DESPAWN_BODY: 2,
} as const;
export type OpCode = (typeof OpCode)[number & keyof typeof OpCode];

// ---- Shape payloads ---------------------------------------------------------

export type Vec3 = { x: number; y: number; z: number };

export type BodySpec = {
  readonly bodyType: "dynamic" | "fixed";
  readonly position: Vec3; // body CENTER
  readonly density?: number;
} & (
  | { readonly kind: "box"; readonly halfExtents: Vec3 }
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

export type WorkerInbound = InitMessage;
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
