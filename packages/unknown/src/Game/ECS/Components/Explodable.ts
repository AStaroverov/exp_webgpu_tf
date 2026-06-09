import { addComponent, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';

export type ExplodableSettings = {
    /** Area damage dealt at the epicenter; falls off linearly to zero at `radius`. */
    damage: number;
    /** Damage radius in world pixels. */
    radius: number;
    /** Explosion VFX sprite size. */
    vfxSize: number;
    /** Light-flash radius. */
    lightRadius: number;
};

/**
 * Marks an entity that detonates when it is destroyed. The explosion is produced
 * uniformly by `createExplodeSystem` whenever the entity also has a `Destroy`
 * component — regardless of what scheduled the destruction (collision, max range,
 * timeout, ...). The settings here drive both the VFX and the area damage.
 */
export const createExplodableComponent = defineComponent((Explodable) => {
    const damage = TypedArray.f64(delegate.defaultSize);
    const radius = TypedArray.f64(delegate.defaultSize);
    const vfxSize = TypedArray.f64(delegate.defaultSize);
    const lightRadius = TypedArray.f64(delegate.defaultSize);
    return {
        damage,
        radius,
        vfxSize,
        lightRadius,
        addComponent(world: World, eid: number, settings: ExplodableSettings) {
            addComponent(world, eid, Explodable);
            damage[eid] = settings.damage;
            radius[eid] = settings.radius;
            vfxSize[eid] = settings.vfxSize;
            lightRadius[eid] = settings.lightRadius;
        },
    };
});
