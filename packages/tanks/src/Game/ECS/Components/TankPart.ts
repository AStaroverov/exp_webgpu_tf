import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { component } from '../../../../../renderer/src/ECS/utils.ts';
import { addComponent, EntityId, removeComponent, World } from 'bitecs';

export const TankPart = component({
    jointPid: TypedArray.f64(delegate.defaultSize),
    
    addComponent(world: World, eid: EntityId, pid: number): void {
        addComponent(world, eid, TankPart);
        TankPart.jointPid[eid] = pid;
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