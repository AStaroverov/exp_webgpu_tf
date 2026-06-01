import { addComponent, EntityId, World } from 'bitecs';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';

// Marker on the hull-brain node identifying it as a tank-style vehicle. The turret is
// found via the Brain hierarchy (hull node's turret child) — see getTurretPhysOfHull.
export const createTankComponent = defineComponent((Tank) => {
    return {
        addComponent(world: World, eid: EntityId) {
            addComponent(world, eid, Tank);
        },
    };
});
