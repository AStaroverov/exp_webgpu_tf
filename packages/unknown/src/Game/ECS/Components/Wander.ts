import { addComponent, EntityId, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';

/**
 * Sinusoidal steering for free-flying bodies (stream particles): each tick the
 * velocity vector is rotated by `sin(age · frequency + phase) · angularSpeed`,
 * so the path meanders along a smooth curve instead of a straight ray.
 * `phase` is seeded at spawn — every particle curves its own way, reproducibly.
 */
export const createWanderComponent = defineComponent((Wander) => {
    /** Per-entity phase offset in radians — decorrelates particles */
    const phase = TypedArray.f32(delegate.defaultSize);
    /** Steering oscillation frequency in rad/ms */
    const frequency = TypedArray.f32(delegate.defaultSize);
    /** Peak turn rate in rad/s */
    const angularSpeed = TypedArray.f32(delegate.defaultSize);
    /** Time since spawn in ms — the argument of the steering sine */
    const ageMs = TypedArray.f64(delegate.defaultSize);
    return {
        phase,
        frequency,
        angularSpeed,
        ageMs,
        addComponent(world: World, eid: EntityId, ph: number, freq: number, turn: number) {
            addComponent(world, eid, Wander);
            phase[eid] = ph;
            frequency[eid] = freq;
            angularSpeed[eid] = turn;
            ageMs[eid] = 0;
        },
    };
});
