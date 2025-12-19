import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { addComponent, World } from 'bitecs';
import { component } from '../../../../../renderer/src/ECS/utils.ts';

/**
 * Component for an exhaust pipe attached to a vehicle.
 * Defines the position and direction of exhaust emissions.
 * Uses Parent component to store the vehicle entity ID.
 */
export const ExhaustPipe = component({
    // Relative position on the vehicle (local coordinates)
    relativeX: TypedArray.f32(delegate.defaultSize),
    relativeY: TypedArray.f32(delegate.defaultSize),
    // Exhaust direction (angle in radians, relative to vehicle rotation)
    direction: TypedArray.f32(delegate.defaultSize),
    // Emission rate (particles per second)
    emissionRate: TypedArray.f32(delegate.defaultSize),
    // Time accumulator for emission timing
    emissionAccumulator: TypedArray.f32(delegate.defaultSize),

    addComponent(
        world: World,
        eid: number,
        relativeX: number,
        relativeY: number,
        direction: number,
        emissionRate: number,
    ) {
        addComponent(world, eid, ExhaustPipe);
        ExhaustPipe.relativeX[eid] = relativeX;
        ExhaustPipe.relativeY[eid] = relativeY;
        ExhaustPipe.direction[eid] = direction;
        ExhaustPipe.emissionRate[eid] = emissionRate;
        ExhaustPipe.emissionAccumulator[eid] = 0;
    },
});

