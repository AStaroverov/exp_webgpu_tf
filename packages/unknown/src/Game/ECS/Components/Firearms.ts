import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { addComponent, EntityId, World } from 'bitecs';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';
import { BulletCaliber } from './Bullet.ts';

export const createFirearmsComponent = defineComponent((Firearms) => {
    const caliber = TypedArray.i8(delegate.defaultSize);
    // Remaining reload ms; the duration itself lives in the caliber's global
    // config and is read at the use site, not copied here. The projectile
    // spawn offset is the separate `SpawnDeltaPosition` component.
    const reloading = TypedArray.f64(delegate.defaultSize);

    return {
        caliber,
        reloading,

        addComponent(world: World, eid: EntityId, cal: BulletCaliber) {
            addComponent(world, eid, Firearms);
            reloading[eid] = 0;
            caliber[eid] = cal;
        },
        isReloading(eid: EntityId): boolean {
            return reloading[eid] > 0;
        },
        startReloading(eid: EntityId, durationMs: number) {
            reloading[eid] = durationMs;
        },
        updateReloading(eid: EntityId, dt: number) {
            reloading[eid] -= dt;
        },
    };
});
