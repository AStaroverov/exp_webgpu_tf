import { hasComponent, query } from "bitecs";
import { GameDI } from "../../../DI/GameDI.ts";
import { ExhaustSmokeOptions, spawnExhaustSmoke } from "../../Entities/ExhaustSmoke.ts";
import {
  GlobalTransform,
  getMatrixTranslationX,
  getMatrixTranslationY,
  getMatrixRotationZ,
} from "../../../../../../renderer/src/ECS/Components/Transform.ts";
import { random } from "../../../../../../../lib/random.ts";
import { hypot } from "../../../../../../../lib/math.ts";
import { getGameComponents } from "../../createGameWorld.ts";

const ACCELERATION_EMISSION_MULTI = 5;
const SMOKE_SIZE_MIN = 3;
const SMOKE_SIZE_MAX = 6;
const SMOKE_VELOCITY_BASE = 15;
const SMOKE_VELOCITY_VARIANCE = 5;

const exhaustSmokeOptions: ExhaustSmokeOptions = {
  x: 0,
  y: 0,
  velocityX: 0,
  velocityY: 0,
  size: 0,
};

export function createExhaustSmokeSpawnSystem({ world } = GameDI) {
  const { ExhaustPipe, Parent, Vehicle, RigidBodyState } = getGameComponents(world);

  return (delta: number) => {
    const pipeEids = query(world, [ExhaustPipe]);
    const deltaSeconds = delta / 1000;

    for (const pipeEid of pipeEids) {
      const vehicleEid = Parent.id.get(pipeEid);

      if (!hasComponent(world, vehicleEid, Vehicle)) continue;

      const vehicleMatrix = GlobalTransform.matrix.getBatch(vehicleEid);
      const vehicleX = getMatrixTranslationX(vehicleMatrix);
      const vehicleY = getMatrixTranslationY(vehicleMatrix);
      const vehicleRotation = getMatrixRotationZ(vehicleMatrix);

      let speed = 0;
      if (hasComponent(world, vehicleEid, RigidBodyState)) {
        const linvelX = RigidBodyState.linvel.get(vehicleEid, 0);
        const linvelY = RigidBodyState.linvel.get(vehicleEid, 1);
        speed = hypot(linvelX, linvelY);
      }

      const speedFactor = Math.min(speed / 50, 1);
      const emissionRate =
        ExhaustPipe.emissionRate.get(pipeEid) * (1 + speedFactor * ACCELERATION_EMISSION_MULTI);

      let acc = ExhaustPipe.emissionAccumulator.get(pipeEid) + deltaSeconds * emissionRate;

      const relX = ExhaustPipe.relativeX.get(pipeEid);
      const relY = ExhaustPipe.relativeY.get(pipeEid);
      const cos = Math.cos(vehicleRotation);
      const sin = Math.sin(vehicleRotation);

      const worldX = vehicleX + relX * cos - relY * sin;
      const worldY = vehicleY + relX * sin + relY * cos;

      const exhaustDir = vehicleRotation + ExhaustPipe.direction.get(pipeEid);

      while (acc >= 1) {
        acc -= 1;

        const velocityMagnitude = SMOKE_VELOCITY_BASE + random() * SMOKE_VELOCITY_VARIANCE;
        const spread = (random() - 0.5) * 0.5;
        const finalDir = exhaustDir + spread;

        const velocityX = Math.cos(finalDir) * velocityMagnitude;
        const velocityY = Math.sin(finalDir) * velocityMagnitude;

        const size = SMOKE_SIZE_MIN + random() * (SMOKE_SIZE_MAX - SMOKE_SIZE_MIN);

        exhaustSmokeOptions.x = worldX;
        exhaustSmokeOptions.y = worldY;
        exhaustSmokeOptions.size = size;
        exhaustSmokeOptions.velocityX = velocityX;
        exhaustSmokeOptions.velocityY = velocityY;
        spawnExhaustSmoke(exhaustSmokeOptions);
      }

      ExhaustPipe.emissionAccumulator.set(pipeEid, acc);
    }
  };
}
