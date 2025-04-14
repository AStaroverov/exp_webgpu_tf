import { delegate } from '../../../../../../src/delegate.ts';
import { NestedArray, TypedArray } from '../../../../../../src/utils.ts';
import { component } from '../../../../../../src/ECS/utils.ts';
import { addComponent, EntityId, World } from 'bitecs';

export const TANK_APPROXIMATE_COLLISION_RADIUS = 80;

export const Tank = component({
    aimEid: TypedArray.f64(delegate.defaultSize),
    turretEId: TypedArray.f64(delegate.defaultSize),
    bulletSpeed: TypedArray.f64(delegate.defaultSize),
    bulletStartPosition: NestedArray.f64(2, delegate.defaultSize),
    initialPartsCount: TypedArray.f64(delegate.defaultSize),

    addComponent(world: World, eid: EntityId, bulletSpeed: number, bulletStartPosition: number[], initialPartsCount: number): void {
        addComponent(world, eid, Tank);
        Tank.bulletSpeed[eid] = bulletSpeed;
        Tank.bulletStartPosition.set(eid, 0, bulletStartPosition[0]);
        Tank.bulletStartPosition.set(eid, 1, bulletStartPosition[1]);
        Tank.initialPartsCount[eid] = initialPartsCount;
    },

    setAimEid(tankEid: number, aimEid: number) {
        Tank.aimEid[tankEid] = aimEid;
    },
    setTurretEid(tankEid: number, turretEid: number) {
        Tank.turretEId[tankEid] = turretEid;
    },
});

