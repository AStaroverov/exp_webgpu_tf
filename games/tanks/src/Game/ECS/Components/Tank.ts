import { delegate } from '../../../../../../src/delegate.ts';
import { NestedArray, TypedArray } from '../../../../../../src/utils.ts';
import { component } from '../../../../../../src/ECS/utils.ts';
import { addComponent, EntityId, World } from 'bitecs';

export const Tank = component({
    turretEId: TypedArray.f64(delegate.defaultSize),

    bulletStartPosition: NestedArray.f64(2, delegate.defaultSize),
    initialPartsCount: TypedArray.f64(delegate.defaultSize),

    addComponent(world: World, eid: EntityId, bulletStartPosition: number[], initialPartsCount: number): void {
        addComponent(world, eid, Tank);
        Tank.bulletStartPosition.set(eid, 0, bulletStartPosition[0]);
        Tank.bulletStartPosition.set(eid, 1, bulletStartPosition[1]);
        Tank.initialPartsCount[eid] = initialPartsCount;
    },

    setTurretEid(tankEid: number, turretEid: number) {
        Tank.turretEId[tankEid] = turretEid;
    },
});

