import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { addComponent, World } from 'bitecs';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';

export const createProgressComponent = defineComponent((Progress) => {
    const age = TypedArray.f32(delegate.defaultSize);
    const maxAge = TypedArray.f32(delegate.defaultSize);
    return {
        age,
        maxAge,
        addComponent(world: World, eid: number, max: number) {
            addComponent(world, eid, Progress);
            age[eid] = 0;
            maxAge[eid] = max;
        },
        updateAge(eid: number, delta: number) {
            age[eid] += delta;
        },
        getProgress(eid: number): number {
            return age[eid] / maxAge[eid];
        },
    };
});
