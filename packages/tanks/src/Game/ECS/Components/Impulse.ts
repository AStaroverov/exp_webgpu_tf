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

/**
 * Maximum number of impulse-at-point entries per entity.
 * Used for tracks/wheels that apply force at specific positions.
 */
const MAX_IMPULSE_POINTS = 4;

/**
 * ImpulseAtPoint component for applying impulses at specific world positions.
 * This creates realistic physics where force applied off-center creates rotation.
 * Multiple impulses can be queued (e.g., for left and right tracks).
 */
export const ImpulseAtPoint = component({
    // Impulse vectors (x, y) for each point
    impulseX: TypedArray.f64(delegate.defaultSize * MAX_IMPULSE_POINTS),
    impulseY: TypedArray.f64(delegate.defaultSize * MAX_IMPULSE_POINTS),
    // World positions where impulses are applied
    pointX: TypedArray.f64(delegate.defaultSize * MAX_IMPULSE_POINTS),
    pointY: TypedArray.f64(delegate.defaultSize * MAX_IMPULSE_POINTS),
    // Number of queued impulses
    count: TypedArray.i8(delegate.defaultSize),

    addComponent(world: World, eid: EntityId) {
        addComponent(world, eid, ImpulseAtPoint);
        ImpulseAtPoint.count[eid] = 0;
    },

    /**
     * Queue an impulse to be applied at a specific world position.
     * @param eid - Entity to apply impulse to
     * @param impulseX - Impulse X component
     * @param impulseY - Impulse Y component  
     * @param worldX - World X position where force is applied
     * @param worldY - World Y position where force is applied
     */
    add(eid: EntityId, impulseX: number, impulseY: number, worldX: number, worldY: number) {
        const idx = ImpulseAtPoint.count[eid];
        if (idx >= MAX_IMPULSE_POINTS) return;
        
        const offset = eid * MAX_IMPULSE_POINTS + idx;
        ImpulseAtPoint.impulseX[offset] = impulseX;
        ImpulseAtPoint.impulseY[offset] = impulseY;
        ImpulseAtPoint.pointX[offset] = worldX;
        ImpulseAtPoint.pointY[offset] = worldY;
        ImpulseAtPoint.count[eid] = idx + 1;
    },

    /**
     * Get impulse data at index.
     */
    get(eid: EntityId, index: number): [impulseX: number, impulseY: number, pointX: number, pointY: number] {
        const offset = eid * MAX_IMPULSE_POINTS + index;
        return [
            ImpulseAtPoint.impulseX[offset],
            ImpulseAtPoint.impulseY[offset],
            ImpulseAtPoint.pointX[offset],
            ImpulseAtPoint.pointY[offset],
        ];
    },

    /**
     * Reset all queued impulses.
     */
    reset(eid: EntityId) {
        ImpulseAtPoint.count[eid] = 0;
    },

    /**
     * Check if there are any impulses to apply.
     */
    hasImpulse(eid: EntityId): boolean {
        return ImpulseAtPoint.count[eid] > 0;
    },
});

