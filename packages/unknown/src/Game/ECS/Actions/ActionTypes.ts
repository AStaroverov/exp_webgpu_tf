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
    Entity = 1,
    Hex = 2,
    Point = 3,
}

/** Action kinds — each kind has a params component and an executor system. */
export enum ActionKind {
    MoveStep = 0,
    Aim = 1,
    Fire = 2,
    Hold = 3,
}

// ── Target addressing — discriminated union over TargetKind ──────────────────

export type ActionNoneTargetSpec = { kind: TargetKind.None };
export type ActionEntityTargetSpec = {
    kind: TargetKind.Entity;
    eid: number;
};
export type ActionHexTargetSpec = { kind: TargetKind.Hex; q: number; r: number };
export type ActionPointTargetSpec = { kind: TargetKind.Point; x: number; y: number };

export type ActionTargetSpec =
    | ActionNoneTargetSpec
    | ActionEntityTargetSpec
    | ActionHexTargetSpec
    | ActionPointTargetSpec;

/** A target that addresses a place in the world (where to aim / which entity). */
export type ActionWorldTargetSpec =
    | ActionEntityTargetSpec
    | ActionHexTargetSpec
    | ActionPointTargetSpec;
