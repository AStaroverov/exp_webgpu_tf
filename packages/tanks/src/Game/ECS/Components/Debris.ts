import { addComponent, World } from 'bitecs';
import { component } from '../../../../../renderer/src/ECS/utils.ts';

/**
 * Debris component marks torn-off tank parts that can be collected by the player.
 * When collected, the debris part is attached back to the player's tank structure.
 */
export const Debris = component({
    addComponent(world: World, eid: number) {
        addComponent(world, eid, Debris);
    },
});
