/**
 * The ActionSchedule lives in its **own** ECS world, separate from the game
 * world. Action entities (Action + ActionTarget + per-kind params) are created
 * here; they only *reference* game-world entities by id (`ownerEid`, an Entity
 * target's `eid`) — the two id spaces are disjoint, same convention as the hex
 * grid's occupancy. Keeping actions in a dedicated world means scheduling never
 * pollutes game queries and can be reset independently.
 */

import { createWorld, World } from 'bitecs';
import { createActionComponent } from './Components/Action.ts';
import { createActionTargetComponent } from './Components/ActionTarget.ts';
import { createMoveToHexParamsComponent } from './Components/params/MoveToHexParams.ts';
import { createWaitParamsComponent } from './Components/params/WaitParams.ts';
import { createTurretAimParamsComponent } from './Components/params/TurretAimParams.ts';
import { createFireParamsComponent } from './Components/params/FireParams.ts';

function createActionComponents(world: World) {
    return {
        Action: createActionComponent(world),
        ActionTarget: createActionTargetComponent(world),
        MoveToHexParams: createMoveToHexParamsComponent(world),
        WaitParams: createWaitParamsComponent(world),
        TurretAimParams: createTurretAimParamsComponent(world),
        FireParams: createFireParamsComponent(world),
    };
}

export type ActionComponents = ReturnType<typeof createActionComponents>;

export type ActionWorld = World<{ components: ActionComponents }>;

export function createActionWorld(): ActionWorld {
    const context = { components: null as unknown as ActionComponents };
    const world = createWorld(context) as ActionWorld;
    context.components = createActionComponents(world);
    return world;
}

export function getActionComponents(world: World): ActionComponents {
    const components = (world as ActionWorld).components;
    if (!components) {
        throw new Error('Action components are not available on this world');
    }
    return components;
}
