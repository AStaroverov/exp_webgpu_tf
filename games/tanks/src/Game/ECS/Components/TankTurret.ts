import { delegate } from '../../../../../../src/delegate.ts';
import { NestedArray, TypedArray } from '../../../../../../src/utils.ts';
import { component } from '../../../../../../src/ECS/utils.ts';
import { addComponent, EntityId, World } from 'bitecs';
import { BulletCaliber } from './Bullet.ts';

export const TankTurret = component({
    tankEId: TypedArray.f64(delegate.defaultSize),

    bulletCaliber: TypedArray.i8(delegate.defaultSize),
    bulletStartPosition: NestedArray.f64(2, delegate.defaultSize),

    addComponent(world: World, eid: EntityId, tankEid: EntityId): void {
        TankTurret.bulletStartPosition.set(eid, 0, 0);
        TankTurret.bulletStartPosition.set(eid, 1, 0);
        TankTurret.bulletCaliber[eid] = 0;

        addComponent(world, eid, TankTurret);
        TankTurret.tankEId[eid] = tankEid;
    },

    setBulletData(eid: number, position: number[], caliber: BulletCaliber): void {
        TankTurret.bulletStartPosition.setBatch(eid, position);
        TankTurret.bulletCaliber[eid] = caliber;
    },
});

