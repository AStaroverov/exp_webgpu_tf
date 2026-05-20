import { delegate } from '../../../../../renderer/src/delegate.ts';
import { NestedArray, TypedArray } from '../../../../../renderer/src/utils.ts';
import { addComponent, EntityId, World } from 'bitecs';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';
import { BulletCaliber } from './Bullet.ts';

export const createFirearmsComponent = defineComponent((Firearms) => {
    const caliber = TypedArray.i8(delegate.defaultSize);
    const bulletStartPosition = NestedArray.f32(2, delegate.defaultSize);
    const reloading = NestedArray.f32(2, delegate.defaultSize);

    return {
        caliber,
        bulletStartPosition,
        reloading,

        addComponent(world: World, eid: EntityId) {
            addComponent(world, eid, Firearms);
            bulletStartPosition.set(eid, 0, 0);
            bulletStartPosition.set(eid, 1, 0);
            reloading.set(eid, 0, 0);
            reloading.set(eid, 1, 0);
            caliber[eid] = 0;
        },
        setData(eid: EntityId, position: number[], cal: BulletCaliber) {
            bulletStartPosition.setBatch(eid, position);
            caliber[eid] = cal;
        },
        setReloadingDuration(eid: EntityId, duration: number) {
            reloading.set(eid, 1, duration);
        },
        isReloading(eid: EntityId): boolean {
            return reloading.get(eid, 0) > 0;
        },
        startReloading(eid: EntityId) {
            const duration = reloading.get(eid, 1);
            reloading.set(eid, 0, duration);
        },
        updateReloading(eid: EntityId, dt: number) {
            const rest = reloading.get(eid, 0);
            reloading.set(eid, 0, rest - dt);
        },
    };
});
