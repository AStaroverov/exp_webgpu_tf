import { addEntity } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { ExhaustPipe } from '../Components/ExhaustPipe.ts';
import { PI } from '../../../../../../lib/math.ts';
import { Parent } from '../Components/Parent.ts';

export interface ExhaustPipeOptions {
    vehicleEid: number;
    relativeX: number;
    relativeY: number;
    direction: number;
    emissionRate?: number;
}

/**
 * Creates an exhaust pipe entity attached to a vehicle.
 * The exhaust pipe will emit smoke particles in the specified direction.
 */
export function createExhaustPipe(options: ExhaustPipeOptions, { world } = GameDI): number {
    const eid = addEntity(world);

    ExhaustPipe.addComponent(
        world,
        eid,
        options.relativeX,
        options.relativeY,
        options.direction,
        options.emissionRate ?? 1,
    );
    Parent.addComponent(world, eid, options.vehicleEid);

    return eid;
}

/**
 * Adds default exhaust pipes to a tank (rear-facing exhausts).
 */
export function addTankExhaustPipes(
    tankEid: number,
    tankWidth: number,
    tankHeight: number,
): [number, number] {
    // Add two exhaust pipes at the rear of the tank, facing backward
    // Front is at -Y (where gun points), rear is at +Y
    const rearY = tankHeight / 2 + 2; // Behind the tank (positive Y is rear)
    const offsetX = tankWidth / 4;

    const leftPipeEid = createExhaustPipe({
        vehicleEid: tankEid,
        relativeX: -offsetX,
        relativeY: rearY,
        direction: PI / 2, // Pointing backward (positive Y direction)
        emissionRate: 5,
    });

    const rightPipeEid = createExhaustPipe({
        vehicleEid: tankEid,
        relativeX: offsetX,
        relativeY: rearY,
        direction: PI / 2,
        emissionRate: 5,
    });

    return [leftPipeEid, rightPipeEid];
}

/**
 * Adds exhaust pipe to a car (single rear exhaust).
 */
export function addCarExhaustPipe(
    carEid: number,
    carWidth: number,
    carHeight: number,
): number {
    // Single exhaust at the rear-right of the car
    // Front is at -Y, rear is at +Y
    const rearY = carHeight / 2 + 2; // Behind the car (positive Y is rear)
    const offsetX = carWidth / 3;

    return createExhaustPipe({
        vehicleEid: carEid,
        relativeX: offsetX,
        relativeY: rearY,
        direction: PI / 2 + 0.2,
        emissionRate: 3, 
    });
}

