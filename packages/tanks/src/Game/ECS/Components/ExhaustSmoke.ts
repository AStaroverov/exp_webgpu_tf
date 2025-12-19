import { addComponent, World } from 'bitecs';
import { component } from '../../../../../renderer/src/ECS/utils.ts';

/**
 * Component marker for exhaust smoke particles.
 * Used by the render system to identify and draw smoke particles.
 */
export const ExhaustSmoke = component({
    addComponent(world: World, eid: number) {
        addComponent(world, eid, ExhaustSmoke);
    },
});

