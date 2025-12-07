/**
 * Tank Movement Sound System
 * Manages Sound component on tanks based on their movement state
 */

import { query, hasComponent } from 'bitecs';
import { GameDI } from '../../../DI/GameDI.ts';
import { Tank, TankType } from '../../Components/Tank.ts';
import { TankController } from '../../Components/TankController.ts';
import { Sound, SoundType, SoundState } from '../../Components/Sound.ts';

const MOVE_THRESHOLD = 0.1;

// Volume based on tank type: Light = 0.6, Medium/Player = 0.8, Heavy = 1.0
const TANK_TYPE_VOLUME: Record<TankType, number> = {
    [TankType.Light]: 0.6,
    [TankType.Medium]: 0.8,
    [TankType.Heavy]: 1.0,
    [TankType.Player]: 0.8,  // Same as Medium
};

/**
 * Check if tank is moving (forward/backward or rotating)
 */
function isTankMoving(eid: number): boolean {
    const move = Math.abs(TankController.move[eid]);
    const rotation = Math.abs(TankController.rotation[eid]);
    return move > MOVE_THRESHOLD || rotation > MOVE_THRESHOLD;
}

/**
 * Get volume based on tank type
 */
function getTankVolume(eid: number): number {
    const tankType = Tank.type[eid] as TankType;
    return TANK_TYPE_VOLUME[tankType] ?? 0.8;
}

/**
 * Create system that manages Sound component for tank movement
 */
export function createTankMoveSoundSystem({ world } = GameDI) {
    return function updateTankMoveSounds(_delta: number): void {
        const tankEids = query(world, [Tank, TankController]);

        for (const eid of tankEids) {
            const isMoving = isTankMoving(eid);
            const hasSound = hasComponent(world, eid, Sound);

            if (isMoving) {
                if (!hasSound) {
                    // Add sound component and start playing
                    Sound.addComponent(world, eid, SoundType.TankMove, {
                        loop: true,
                        volume: getTankVolume(eid),
                        autoplay: true,
                    });
                } else if (Sound.type[eid] === SoundType.TankMove && Sound.state[eid] !== SoundState.Playing) {
                    // Resume if paused
                    Sound.play(eid);
                }
            } else {
                if (hasSound && Sound.type[eid] === SoundType.TankMove) {
                    // Stop movement sound when not moving
                    Sound.stop(eid);
                }
            }
        }
    };
}
