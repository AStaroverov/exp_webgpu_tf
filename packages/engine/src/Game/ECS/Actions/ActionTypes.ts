/**
 * ActionSchedule — enums + target-addressing spec types.
 *
 * Actions live on the controlled entity as the `ActionsQueue` component (slot 0 =
 * front). See `Actions/QUEUE_REFACTOR_PLAN.md`.
 */

export enum ActionStatus {
  Idle = 0,
  Running = 1,
  Finished = 2,
}

/** What an action is addressed to. */
export enum TargetKind {
  None = 0,
  Hex = 1,
}

/** Action kinds — each kind has a params component and an executor system. */
export enum ActionKind {
  MoveStep = 0,
  Aim = 1,
  Fire = 2,
  Hold = 3,
  FireStream = 4,
}

// ── Target addressing — discriminated union over TargetKind ──────────────────

export type ActionNoneTargetSpec = { kind: TargetKind.None };
export type ActionHexTargetSpec = { kind: TargetKind.Hex; q: number; r: number };

export type ActionTargetSpec = ActionNoneTargetSpec | ActionHexTargetSpec;
