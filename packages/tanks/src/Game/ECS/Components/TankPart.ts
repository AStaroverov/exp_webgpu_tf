import { NestedArray, TypedArray } from '../../../../../renderer/src/utils.ts';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { component } from '../../../../../renderer/src/ECS/utils.ts';
import { addComponent, EntityId, removeComponent, World } from 'bitecs';
import { Vector2 } from '@dimforge/rapier2d-simd';

export const TankPart = component({
    jointPid: TypedArray.f64(delegate.defaultSize),
    anchor1: NestedArray.f64(2, delegate.defaultSize),
    anchor2: NestedArray.f64(2, delegate.defaultSize),

    addComponent(world: World, eid: EntityId, pid: number, anchor1: Vector2, anchro2: Vector2): void {
        addComponent(world, eid, TankPart);
        TankPart.jointPid[eid] = pid;
        TankPart.anchor1.set(eid, 0, anchor1.x);
        TankPart.anchor1.set(eid, 1, anchor1.y);
        TankPart.anchor2.set(eid, 0, anchro2.x);
        TankPart.anchor2.set(eid, 1, anchro2.y);
    },
    resetComponent(eid: EntityId): void {
        TankPart.jointPid[eid] = -1;
    },
});

export const TankPartCaterpillar = {
    removeComponent(world: World, eid: EntityId) {
        removeComponent(world, eid, TankPartCaterpillar);
    },
};