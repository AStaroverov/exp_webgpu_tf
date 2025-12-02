import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { addComponent, World } from 'bitecs';
import { component } from '../../../../../renderer/src/ECS/utils.ts';

export const TankTrack = component({
    age: TypedArray.f32(delegate.defaultSize),
    maxAge: TypedArray.f32(delegate.defaultSize),

    addComponent(world: World, eid: number, maxAge: number) {
        addComponent(world, eid, TankTrack);
        TankTrack.age[eid] = 0;
        TankTrack.maxAge[eid] = maxAge;
    },

    updateAge(eid: number, delta: number) {
        TankTrack.age[eid] += delta;
    },

    getProgress(eid: number): number {
        return TankTrack.age[eid] / TankTrack.maxAge[eid];
    },
});
