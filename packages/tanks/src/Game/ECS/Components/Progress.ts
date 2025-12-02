import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { addComponent, World } from 'bitecs';
import { component } from '../../../../../renderer/src/ECS/utils.ts';

export const Progress = component({
    age: TypedArray.f32(delegate.defaultSize),
    maxAge: TypedArray.f32(delegate.defaultSize),

    addComponent(world: World, eid: number, maxAge: number) {
        addComponent(world, eid, Progress);
        Progress.age[eid] = 0;
        Progress.maxAge[eid] = maxAge;
    },

    updateAge(eid: number, delta: number) {
        Progress.age[eid] += delta;
    },

    getProgress(eid: number): number {
        return Progress.age[eid] / Progress.maxAge[eid];
    },
});
