import { query, hasComponent } from "bitecs";
import { GameDI } from "../../../DI/GameDI.ts";
import { SoundType, SoundState } from "../../Components/Sound.ts";
import { getGameComponents } from "../../createGameWorld.ts";

const MOVE_THRESHOLD = 0.1;

export function createTankMoveSoundSystem({ world } = GameDI) {
  const { Sound, Parent, SoundParentRelative, Vehicle, VehicleController } =
    getGameComponents(world);

  function isVehicleMoving(vehicleEid: number): boolean {
    const move = Math.abs(VehicleController.move[vehicleEid]);
    const rotation = Math.abs(VehicleController.rotation[vehicleEid]);
    return move > MOVE_THRESHOLD || rotation > MOVE_THRESHOLD;
  }

  return function updateVehicleMoveSounds(_delta: number): void {
    const soundEids = query(world, [Sound, Parent, SoundParentRelative]);

    for (const soundEid of soundEids) {
      if (Sound.type[soundEid] !== SoundType.TankMove) continue;

      const parentEid = Parent.id[soundEid];

      if (!hasComponent(world, parentEid, Vehicle)) {
        continue;
      }

      const isMoving = isVehicleMoving(parentEid);
      const isPlaying = Sound.state[soundEid] === SoundState.Playing;

      if (isMoving && !isPlaying) {
        Sound.play(soundEid);
      } else if (!isMoving && isPlaying) {
        Sound.stop(soundEid);
      }
    }
  };
}
