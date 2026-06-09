import { addComponent, World } from 'bitecs';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';

/**
 * Marks a vehicle that aims its weapon by rotating its **hull** instead of its
 * turret (the turret is fixed). The Aim/Fire executors query for it: when
 * present, they steer `VehicleController` toward the target using the hull's
 * heading rather than nudging `TurretController`. Rocket tanks carry it — their
 * launcher is bolted to the body, so the whole vehicle turns to point.
 */
export const createHullAimedComponent = defineComponent((HullAimed) => {
    return {
        addComponent(world: World, eid: number) {
            addComponent(world, eid, HullAimed);
        },
    };
});
