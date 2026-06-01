import { query, hasComponent } from 'bitecs';
import { SoundType, SoundState } from '../../Components/Sound.ts';
import { getSoundWorldComponents } from '../../createSoundWorld.ts';
import { getBrainWorldComponents } from '../../createBrainWorld.ts';
import { getNodeByPhysics, getSoundOwnerOf } from '../../refs.ts';
import { Worlds } from '../../../DI/Worlds.ts';

const MOVE_THRESHOLD = 0.1;

export function createTankMoveSoundSystem({ soundWorld, brainWorld } = Worlds) {
    const { Vehicle, VehicleController } = getBrainWorldComponents(brainWorld);

    function isVehicleMoving(brainEid: number): boolean {
        const move = Math.abs(VehicleController.move[brainEid]);
        const rotation = Math.abs(VehicleController.rotation[brainEid]);
        return move > MOVE_THRESHOLD || rotation > MOVE_THRESHOLD;
    }

    return function updateVehicleMoveSounds(_delta: number): void {
        const { Sound, SoundParentRelative } = getSoundWorldComponents(soundWorld);
        const soundEids = query(soundWorld, [Sound, SoundParentRelative]);

        for (const soundEid of soundEids) {
            if (Sound.type[soundEid] !== SoundType.TankMove) continue;

            const ownerAtomEid = getSoundOwnerOf(soundEid);
            // sound -> owner atom (hull) -> the brain node whose presentation is that atom.
            const ownerBrain = getNodeByPhysics(ownerAtomEid);

            if (!hasComponent(brainWorld, ownerBrain, Vehicle)) {
                continue;
            }

            const isMoving = isVehicleMoving(ownerBrain);
            const isPlaying = Sound.state[soundEid] === SoundState.Playing;

            if (isMoving && !isPlaying) {
                Sound.play(soundEid);
            } else if (!isMoving && isPlaying) {
                Sound.stop(soundEid);
            }
        }
    };
}
