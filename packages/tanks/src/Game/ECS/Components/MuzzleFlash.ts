import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { addComponent, World } from 'bitecs';
import { component } from '../../../../../renderer/src/ECS/utils.ts';

export const MuzzleFlash = component({
    x: TypedArray.f32(delegate.defaultSize),
    y: TypedArray.f32(delegate.defaultSize),
    size: TypedArray.f32(delegate.defaultSize),
    age: TypedArray.f32(delegate.defaultSize),
    maxAge: TypedArray.f32(delegate.defaultSize),
    rotation: TypedArray.f32(delegate.defaultSize),

    addComponent(world: World, eid: number, x: number, y: number, size: number, maxAge: number, rotation: number = 0) {
        addComponent(world, eid, MuzzleFlash);
        MuzzleFlash.x[eid] = x;
        MuzzleFlash.y[eid] = y;
        MuzzleFlash.size[eid] = size;
        MuzzleFlash.age[eid] = 0;
        MuzzleFlash.maxAge[eid] = maxAge;
        MuzzleFlash.rotation[eid] = rotation;
    },

    updateAge(eid: number, delta: number) {
        MuzzleFlash.age[eid] += delta;
    },

    getProgress(eid: number): number {
        return MuzzleFlash.age[eid] / MuzzleFlash.maxAge[eid];
    },
});
