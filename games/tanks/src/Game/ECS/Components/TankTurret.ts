import { delegate } from '../../../../../../src/delegate.ts';
import { NestedArray, TypedArray } from '../../../../../../src/utils.ts';
import { component } from '../../../../../../src/ECS/utils.ts';
import { addComponent, EntityId, World } from 'bitecs';
import { BulletCaliber } from './Bullet.ts';

export const TankTurret = component({
    tankEId: TypedArray.f64(delegate.defaultSize),

    bulletCaliber: TypedArray.i8(delegate.defaultSize),
    // [x,y]
    bulletStartPosition: NestedArray.f32(2, delegate.defaultSize),
    // [rest, duration]
    bulletReloading: NestedArray.f32(2, delegate.defaultSize),

    addComponent(world: World, eid: EntityId, tankEid: EntityId): void {
        TankTurret.bulletStartPosition.set(eid, 0, 0);
        TankTurret.bulletStartPosition.set(eid, 1, 0);
        TankTurret.bulletReloading.set(eid, 0, 0);
        TankTurret.bulletReloading.set(eid, 1, 0);
        TankTurret.bulletCaliber[eid] = 0;

        addComponent(world, eid, TankTurret);
        TankTurret.tankEId[eid] = tankEid;
    },

    setBulletData(eid: number, position: number[], caliber: BulletCaliber): void {
        TankTurret.bulletStartPosition.setBatch(eid, position);
        TankTurret.bulletCaliber[eid] = caliber;
    },

    setReloadingDuration(eid: number, duration: number): void {
        TankTurret.bulletReloading.set(eid, 1, duration);
    },
    isReloading(eid: number): boolean {
        return TankTurret.bulletReloading.get(eid, 0) > 0;
    },
    startReloading: ((eid: number): void => {
        const duration = TankTurret.bulletReloading.get(eid, 1);
        TankTurret.bulletReloading.set(eid, 0, duration);
    }),
    updateReloading: ((eid: number, dt: number): void => {
        const rest = TankTurret.bulletReloading.get(eid, 0);
        TankTurret.bulletReloading.set(eid, 0, rest - dt);
    }),
});

