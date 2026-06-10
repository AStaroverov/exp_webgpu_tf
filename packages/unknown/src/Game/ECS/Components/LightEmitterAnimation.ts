import { addComponent, EntityId, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';

/**
 * Animation track for a `LightEmitter`: the emitter's PEAK params — its values
 * at progress 0. Presence (together with `Progress` as the clock) = "this light
 * decays over its lifetime": the animation system rescales the live emitter
 * from these each frame, so the live value never accumulates error.
 */
export const createLightEmitterAnimationComponent = defineComponent((LightEmitterAnimation) => {
    const intensity = TypedArray.f64(delegate.defaultSize);
    const radius = TypedArray.f64(delegate.defaultSize);
    return {
        intensity,
        radius,
        addComponent(world: World, eid: EntityId, i: number, r = 0) {
            addComponent(world, eid, LightEmitterAnimation);
            intensity[eid] = i;
            radius[eid] = r;
        },
    };
});
