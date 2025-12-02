import { addComponent, World } from 'bitecs';
import { component } from '../../../../../renderer/src/ECS/utils.ts';

export const Explosion = component({
    addComponent(world: World, eid: number) {
        addComponent(world, eid, Explosion);
    },
});
