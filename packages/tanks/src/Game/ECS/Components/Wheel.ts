import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { component } from '../../../../../renderer/src/ECS/utils.ts';
import { addComponent, EntityId, removeComponent, World } from 'bitecs';

export enum WheelPosition {
    FrontLeft = 0,
    FrontRight = 1,
    RearLeft = 2,
    RearRight = 3,
}

/**
 * Component for a wheel entity.
 * Each wheel is an independent unit attached to the vehicle via joint.
 */
export const Wheel = component({
    addComponent(world: World, eid: EntityId): void {
        addComponent(world, eid, Wheel);
    }
});

/**
 * Tag component for wheels that can steer (typically front wheels).
 */
export const WheelSteerable = component({
    // Maximum steering angle in radians
    maxSteeringAngle: TypedArray.f64(delegate.defaultSize),
    // Steering speed (radians per second)
    steeringSpeed: TypedArray.f64(delegate.defaultSize),

    addComponent(world: World, eid: EntityId, maxAngle: number = Math.PI / 6, steeringSpeed: number = Math.PI * 2): void {
        addComponent(world, eid, WheelSteerable);
        WheelSteerable.maxSteeringAngle[eid] = maxAngle;
        WheelSteerable.steeringSpeed[eid] = steeringSpeed;
    },

    removeComponent(world: World, eid: EntityId): void {
        removeComponent(world, eid, WheelSteerable);
    },
});

/**
 * Tag component for wheels that receive drive power (typically rear wheels or all-wheel drive).
 */
export const WheelDrive = component({
    addComponent(world: World, eid: EntityId): void {
        addComponent(world, eid, WheelDrive);
    },

    removeComponent(world: World, eid: EntityId): void {
        removeComponent(world, eid, WheelDrive);
    },
});

