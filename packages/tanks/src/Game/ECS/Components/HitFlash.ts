import { addComponent, World } from 'bitecs';
import { component } from '../../../../../renderer/src/ECS/utils.ts';

export const HitFlash = component({
    addComponent(world: World, eid: number) {
        addComponent(world, eid, HitFlash);
    },
});
