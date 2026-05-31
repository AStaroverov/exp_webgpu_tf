import { query, hasComponent } from 'bitecs';
import { SoundType, SoundState } from '../../Components/Sound.ts';
import { getPhysicsWorldComponents } from '../../createPhysicsWorld.ts';
import { getRenderWorldComponents } from '../../createRenderWorld.ts';
import { BridgeDI } from '../../../DI/BridgeDI.ts';
import { Worlds } from '../../../DI/Worlds.ts';

const MOVE_THRESHOLD = 0.1;

export function createTankMoveSoundSystem({ physicsWorld, renderWorld } = Worlds) {
    const { Vehicle, VehicleController } = getPhysicsWorldComponents(physicsWorld);

    function isVehicleMoving(vehiclePhysEid: number): boolean {
        const move = Math.abs(VehicleController.move[vehiclePhysEid]);
        const rotation = Math.abs(VehicleController.rotation[vehiclePhysEid]);
        return move > MOVE_THRESHOLD || rotation > MOVE_THRESHOLD;
    }

    return function updateVehicleMoveSounds(_delta: number): void {
        const { Sound, Parent, SoundParentRelative } = getRenderWorldComponents(renderWorld);
        const soundEids = query(renderWorld, [Sound, Parent, SoundParentRelative]);

        for (const soundEid of soundEids) {
            if (Sound.type[soundEid] !== SoundType.TankMove) continue;

            const parentRenderEid = Parent.id[soundEid];
            const parentPhysEid = BridgeDI.getPhysicsOf(parentRenderEid);

            if (!hasComponent(physicsWorld, parentPhysEid, Vehicle)) {
                continue;
            }

            const isMoving = isVehicleMoving(parentPhysEid);
            const isPlaying = Sound.state[soundEid] === SoundState.Playing;

            if (isMoving && !isPlaying) {
                Sound.play(soundEid);
            } else if (!isMoving && isPlaying) {
                Sound.stop(soundEid);
            }
        }
    };
}
