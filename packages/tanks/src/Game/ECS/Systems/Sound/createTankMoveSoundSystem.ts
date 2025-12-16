/**
 * Tank Movement Sound System
 * Controls tank movement sounds based on parent tank's TankController state
 * 
 * Sound entities are created in createTankBase with:
 * - Parent component pointing to tank
 * - SoundParentRelative marker
 * - SoundType.TankMove
 */

import { query, hasComponent } from 'bitecs';
import { GameDI } from '../../../DI/GameDI.ts';
import { Tank } from '../../Components/Tank.ts';
import { TankController } from '../../Components/TankController.ts';
import { Sound, SoundType, SoundState, SoundParentRelative } from '../../Components/Sound.ts';
import { Parent } from '../../Components/Parent.ts';

const MOVE_THRESHOLD = 0.1;

/**
 * Check if tank is moving (forward/backward or rotating)
 */
function isTankMoving(tankEid: number): boolean {
    const move = Math.abs(TankController.move[tankEid]);
    const rotation = Math.abs(TankController.rotation[tankEid]);
    return move > MOVE_THRESHOLD || rotation > MOVE_THRESHOLD;
}

/**
 * Create system that controls tank movement sounds
 * Queries sound entities with SoundParentRelative and checks if parent is a tank
 */
export function createTankMoveSoundSystem({ world } = GameDI) {
    return function updateTankMoveSounds(_delta: number): void {
        // Query all sound entities that follow a parent
        const soundEids = query(world, [Sound, Parent, SoundParentRelative]);

        for (const soundEid of soundEids) {
            // Only handle TankMove sounds
            if (Sound.type[soundEid] !== SoundType.TankMove) continue;

            const parentEid = Parent.id[soundEid];

            // Check if parent is a tank with controller
            if (!hasComponent(world, parentEid, Tank) || !hasComponent(world, parentEid, TankController)) {
                continue;
            }

            const isMoving = isTankMoving(parentEid);
            const isPlaying = Sound.state[soundEid] === SoundState.Playing;

            if (isMoving && !isPlaying) {
                // Start playing when tank moves
                Sound.play(soundEid);
            } else if (!isMoving && isPlaying) {
                // Stop when tank stops
                Sound.stop(soundEid);
            }
        }
    };
}
