import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { addComponent, World } from 'bitecs';
import { component } from '../../../../../renderer/src/ECS/utils.ts';

export const Explosion = component({
    x: TypedArray.f32(delegate.defaultSize),
    y: TypedArray.f32(delegate.defaultSize),
    size: TypedArray.f32(delegate.defaultSize),
    age: TypedArray.f32(delegate.defaultSize),
    maxAge: TypedArray.f32(delegate.defaultSize),

    addComponent(world: World, eid: number, x: number, y: number, size: number, maxAge: number) {
        addComponent(world, eid, Explosion);
        Explosion.x[eid] = x;
        Explosion.y[eid] = y;
        Explosion.size[eid] = size;
        Explosion.age[eid] = 0;
        Explosion.maxAge[eid] = maxAge;
    },

    updateAge(eid: number, delta: number) {
        Explosion.age[eid] += delta;
    },

    getProgress(eid: number): number {
        return Explosion.age[eid] / Explosion.maxAge[eid];
    },
});
