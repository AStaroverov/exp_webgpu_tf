import { addComponent, EntityId, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { component } from '../../../../../renderer/src/ECS/utils.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';

/**
 * Impulse component for applying linear impulses to rigid bodies.
 * Impulses are accumulated during a frame and applied by the impulse system.
 * After applying, the values are reset to 0.
 */
export const Impulse = component({
    x: TypedArray.f64(delegate.defaultSize),
    y: TypedArray.f64(delegate.defaultSize),

    addComponent(world: World, eid: EntityId) {
        addComponent(world, eid, Impulse);
        Impulse.x[eid] = 0;
        Impulse.y[eid] = 0;
    },

    /**
     * Add impulse to the entity. Impulses accumulate during a frame.
     */
    add(eid: EntityId, x: number, y: number) {
        Impulse.x[eid] += x;
        Impulse.y[eid] += y;
    },

    /**
     * Set impulse directly (replaces any accumulated impulse).
     */
    set(eid: EntityId, x: number, y: number) {
        Impulse.x[eid] = x;
        Impulse.y[eid] = y;
    },

    /**
     * Reset impulse to zero.
     */
    reset(eid: EntityId) {
        Impulse.x[eid] = 0;
        Impulse.y[eid] = 0;
    },

    /**
     * Check if there's any impulse to apply.
     */
    hasImpulse(eid: EntityId): boolean {
        return Impulse.x[eid] !== 0 || Impulse.y[eid] !== 0;
    },
});

/**
 * TorqueImpulse component for applying angular impulses to rigid bodies.
 * Impulses are accumulated during a frame and applied by the impulse system.
 * After applying, the value is reset to 0.
 */
export const TorqueImpulse = component({
    value: TypedArray.f64(delegate.defaultSize),

    addComponent(world: World, eid: EntityId) {
        addComponent(world, eid, TorqueImpulse);
        TorqueImpulse.value[eid] = 0;
    },

    /**
     * Add torque impulse to the entity. Impulses accumulate during a frame.
     */
    add(eid: EntityId, torque: number) {
        TorqueImpulse.value[eid] += torque;
    },

    /**
     * Set torque impulse directly (replaces any accumulated impulse).
     */
    set(eid: EntityId, torque: number) {
        TorqueImpulse.value[eid] = torque;
    },

    /**
     * Reset torque impulse to zero.
     */
    reset(eid: EntityId) {
        TorqueImpulse.value[eid] = 0;
    },

    /**
     * Check if there's any torque impulse to apply.
     */
    hasImpulse(eid: EntityId): boolean {
        return TorqueImpulse.value[eid] !== 0;
    },
});

