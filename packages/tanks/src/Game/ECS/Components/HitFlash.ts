import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { addComponent, World } from 'bitecs';
import { component } from '../../../../../renderer/src/ECS/utils.ts';

export const HitFlash = component({
    x: TypedArray.f32(delegate.defaultSize),
    y: TypedArray.f32(delegate.defaultSize),
    size: TypedArray.f32(delegate.defaultSize),
    age: TypedArray.f32(delegate.defaultSize),
    maxAge: TypedArray.f32(delegate.defaultSize),

    addComponent(world: World, eid: number, x: number, y: number, size: number, maxAge: number) {
        addComponent(world, eid, HitFlash);
        HitFlash.x[eid] = x;
        HitFlash.y[eid] = y;
        HitFlash.size[eid] = size;
        HitFlash.age[eid] = 0;
        HitFlash.maxAge[eid] = maxAge;
    },

    updateAge(eid: number, delta: number) {
        HitFlash.age[eid] += delta;
    },

    getProgress(eid: number): number {
        return HitFlash.age[eid] / HitFlash.maxAge[eid];
    },
});
