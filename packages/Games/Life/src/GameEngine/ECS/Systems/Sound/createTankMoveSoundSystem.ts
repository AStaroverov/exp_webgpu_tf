/**
 * Vehicle Movement Sound System
 * Controls vehicle movement sounds based on parent vehicle's VehicleController state
 * 
 * Sound entities are created in createVehicleBase with:
 * - Parent component pointing to vehicle
 * - SoundParentRelative marker
 * - SoundType.TankMove
 */

import { query, hasComponent } from 'bitecs';
import { GameDI } from '../../../DI/GameDI.js';
import { Vehicle } from '../../Components/Vehicle.js';
import { VehicleController } from '../../Components/VehicleController.js';
import { Sound, SoundType, SoundState, SoundParentRelative } from '../../Components/Sound.js';
import { Parent } from '../../Components/Parent.js';

const MOVE_THRESHOLD = 0.1;

/**
 * Check if vehicle is moving (forward/backward or rotating)
 */
function isVehicleMoving(vehicleEid: number): boolean {
    const move = Math.abs(VehicleController.move[vehicleEid]);
    const rotation = Math.abs(VehicleController.rotation[vehicleEid]);
    return move > MOVE_THRESHOLD || rotation > MOVE_THRESHOLD;
}

/**
 * Create system that controls vehicle movement sounds
 * Queries sound entities with SoundParentRelative and checks if parent is a vehicle
 */
export function createTankMoveSoundSystem({ world } = GameDI) {
    return function updateVehicleMoveSounds(_delta: number): void {
        // Query all sound entities that follow a parent
        const soundEids = query(world, [Sound, Parent, SoundParentRelative]);

        for (const soundEid of soundEids) {
            // Only handle TankMove sounds
            if (Sound.type[soundEid] !== SoundType.TankMove) continue;

            const parentEid = Parent.id[soundEid];

            // Check if parent is a vehicle
            if (!hasComponent(world, parentEid, Vehicle)) {
                continue;
            }

            const isMoving = isVehicleMoving(parentEid);
            const isPlaying = Sound.state[soundEid] === SoundState.Playing;

            if (isMoving && !isPlaying) {
                // Start playing when vehicle moves
                Sound.play(soundEid);
            } else if (!isMoving && isPlaying) {
                // Stop when vehicle stops
                Sound.stop(soundEid);
            }
        }
    };
}
