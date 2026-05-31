/**
 * Action registry — the single source of truth mapping each `ActionKind` to its
 * descriptor (each co-located with its executor system). `EnqueueActionSpec` is
 * *derived* from the registered descriptors, so adding a kind is one entry here
 * (plus its system + descriptor file) — no hand-maintained union, no switches.
 */

import { ActionKind } from './ActionTypes.ts';
import { Worlds } from '../../DI/Worlds.ts';
import { MoveToHexActionDescriptor } from './systems/createMoveToHexActionSystem.ts';
import { WaitActionDescriptor } from './systems/createWaitActionSystem.ts';
import { TurretAimActionDescriptor } from './systems/createTurretAimActionSystem.ts';
import { FireActionDescriptor } from './systems/createFireActionSystem.ts';

export const ACTION_REGISTRY = {
    [ActionKind.MoveToHex]: MoveToHexActionDescriptor,
    [ActionKind.Wait]: WaitActionDescriptor,
    [ActionKind.TurretAim]: TurretAimActionDescriptor,
    [ActionKind.Fire]: FireActionDescriptor,
};

type Registry = typeof ACTION_REGISTRY;

/** Enqueue spec union — derived from every registered descriptor's `createAction`. */
export type EnqueueActionSpec = Parameters<Registry[keyof Registry]['createAction']>[2];

/**
 * Build all registered executor systems once and return a function that runs
 * them all per tick. Each executor decides for itself whether the global top
 * action is its responsibility.
 */
export function createRunExecutors({ actionWorld, physicsWorld } = Worlds): (delta: number) => void {
    const systems = Object.values(ACTION_REGISTRY).map((d) => d.createSystem(actionWorld, physicsWorld));

    return function runExecutors(delta: number) {
        for (let i = 0; i < systems.length; i++) {
            systems[i](delta);
        }
    };
}
