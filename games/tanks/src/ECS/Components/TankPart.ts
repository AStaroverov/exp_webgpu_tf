import { TypedArray } from '../../../../../src/utils.ts';
import { delegate } from '../../../../../src/delegate.ts';
import { component } from '../../../../../src/ECS/utils.ts';
import { addComponent, EntityId, World } from 'bitecs';

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
