import { addComponent, World } from 'bitecs';
import { component } from '../../../../../renderer/src/ECS/utils.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { delegate } from '../../../../../renderer/src/delegate.ts';

/**
 * VFX effect types - must match shader constants
 */
export const VFXType = {
    ExhaustSmoke: 0,
    Explosion: 1,
    HitFlash: 2,
    MuzzleFlash: 3,
} as const;

export type VFXTypeValue = (typeof VFXType)[keyof typeof VFXType];

/**
 * Component for visual effects (explosions, flashes, smoke, etc.)
 */
export const VFX = component({
    type: TypedArray.u8(delegate.defaultSize),

    addComponent(world: World, eid: number, type: VFXTypeValue) {
        addComponent(world, eid, VFX);
        VFX.type[eid] = type;
    },
});

