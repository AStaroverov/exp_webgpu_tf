import { addComponent, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';

export const createDamagableComponent = defineComponent((Damagable) => {
    const damage = TypedArray.f64(delegate.defaultSize);
    return {
        damage,
        addComponent(world: World, eid: number, dmg: number) {
            addComponent(world, eid, Damagable);
            damage[eid] = dmg;
        },
    };
});
