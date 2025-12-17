import { delegate } from '../../../../../renderer/src/delegate.ts';
import { NestedArray, TypedArray } from '../../../../../renderer/src/utils.ts';
import { component } from '../../../../../renderer/src/ECS/utils.ts';
import { addComponent, EntityId, World } from 'bitecs';
import { BulletCaliber } from './Bullet.ts';

export const Firearms = component({
    caliber: TypedArray.i8(delegate.defaultSize),
    // [x, y]
    bulletStartPosition: NestedArray.f32(2, delegate.defaultSize),
    // [rest, duration]
    reloading: NestedArray.f32(2, delegate.defaultSize),

    addComponent(world: World, eid: EntityId): void {
        addComponent(world, eid, Firearms);
        Firearms.bulletStartPosition.set(eid, 0, 0);
        Firearms.bulletStartPosition.set(eid, 1, 0);
        Firearms.reloading.set(eid, 0, 0);
        Firearms.reloading.set(eid, 1, 0);
        Firearms.caliber[eid] = 0;
    },

    setData(eid: EntityId, position: number[], caliber: BulletCaliber): void {
        Firearms.bulletStartPosition.setBatch(eid, position);
        Firearms.caliber[eid] = caliber;
    },

    setReloadingDuration(eid: EntityId, duration: number): void {
        Firearms.reloading.set(eid, 1, duration);
    },

    isReloading(eid: EntityId): boolean {
        return Firearms.reloading.get(eid, 0) > 0;
    },

    startReloading(eid: EntityId): void {
        const duration = Firearms.reloading.get(eid, 1);
        Firearms.reloading.set(eid, 0, duration);
    },

    updateReloading(eid: EntityId, dt: number): void {
        const rest = Firearms.reloading.get(eid, 0);
        Firearms.reloading.set(eid, 0, rest - dt);
    },
});

