import { query } from "bitecs";
import { GameDI } from "../../../DI/GameDI.ts";
import { abs } from "../../../../../../../lib/math.ts";
import { random } from "../../../../../../../lib/random.ts";
import { spawnTreadMark } from "../../Entities/TreadMark.ts";
import {
  GlobalTransform,
  getMatrixTranslationX,
  getMatrixTranslationY,
  getMatrixRotationZ,
} from "../../../../../../renderer/src/ECS/Components/Transform.ts";
import { getGameComponents } from "../../createGameWorld.ts";

const BASE_CHANCE = 0.01;
const SKID_CHANCE = 0.5;
const ROTATION_MULTIPLIER = 0.3;

const TREAD_MARK_WIDTH = 2.5;
const TREAD_MARK_HEIGHT = 4;

export function createSpawnWheelTreadMarksSystem({ world } = GameDI) {
  const { Wheel, RigidBodyState, Impulse } = getGameComponents(world);

  return (_delta: number) => {
    const wheelEids = query(world, [Wheel, RigidBodyState, Impulse]);

    for (const eid of wheelEids) {
      const linvel = RigidBodyState.linvel.getBatch(eid);
      const rotation = RigidBodyState.rotation[eid];

      const impulseX = Impulse.x.get(eid);
      const impulseY = Impulse.y.get(eid);

      let chance = BASE_CHANCE;

      const dotProduct = linvel[0] * impulseX + linvel[1] * impulseY;
      if (dotProduct <= 0) {
        chance = Math.max(chance, SKID_CHANCE);
      }

      if (rotation !== 0) {
        chance = Math.max(chance, abs(rotation) * ROTATION_MULTIPLIER);
      }

      if (random() > chance) continue;

      const globalMatrix = GlobalTransform.matrix.getBatch(eid);
      const x = getMatrixTranslationX(globalMatrix);
      const y = getMatrixTranslationY(globalMatrix);
      const rot = getMatrixRotationZ(globalMatrix);

      spawnTreadMark({
        x,
        y,
        width: TREAD_MARK_WIDTH,
        height: TREAD_MARK_HEIGHT,
        rotation: rot,
      });
    }
  };
}
