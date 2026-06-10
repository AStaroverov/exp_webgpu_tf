/**
 * Action registry — the single source of truth mapping each `ActionKind` to its
 * descriptor (each co-located with its executor system). `EnqueueActionSpec` is
 * *derived* from the registered descriptors, so adding a kind is one entry here
 * (plus its system + descriptor file) — no hand-maintained union, no switches.
 */

import { ActionKind } from './ActionTypes.ts';
import { MoveStepActionDescriptor } from './systems/MoveStepAction.ts';
import { HoldActionDescriptor } from './systems/HoldAction.ts';
import { AimActionDescriptor } from './systems/AimAction.ts';
import { FireActionDescriptor } from './systems/FireAction.ts';
import { FireStreamActionDescriptor } from './systems/FireStreamAction.ts';

export const ACTION_REGISTRY = {
    [ActionKind.MoveStep]: MoveStepActionDescriptor,
    [ActionKind.Hold]: HoldActionDescriptor,
    [ActionKind.Aim]: AimActionDescriptor,
    [ActionKind.Fire]: FireActionDescriptor,
    [ActionKind.FireStream]: FireStreamActionDescriptor,
};

type Registry = typeof ACTION_REGISTRY;

/** Enqueue spec union — derived from every registered descriptor's `encode`. */
export type EnqueueActionSpec = Parameters<Registry[keyof Registry]['encode']>[2];

/**
 * Build all registered executor systems once and return a function that runs
 * them all per tick. Each executor decides for itself whether each owner's front
 * action is its responsibility.
 */
export function createRunExecutors(): (delta: number) => void {
    const systems = Object.values(ACTION_REGISTRY).map((d) => d.createSystem());

    return function runExecutors(delta: number) {
        for (let i = 0; i < systems.length; i++) {
            systems[i](delta);
        }
    };
}
