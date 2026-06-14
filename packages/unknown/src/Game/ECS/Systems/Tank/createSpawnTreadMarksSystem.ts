import { query } from "bitecs";
import { GameDI } from "../../../DI/GameDI.ts";
import { abs } from "../../../../../../../lib/math.ts";
import { random } from "../../../../../../../lib/random.ts";
import { spawnTreadMark, TreadMarkOptions } from "../../Entities/TreadMark.ts";
import {
  GlobalTransform,
  getMatrixTranslationX,
  getMatrixTranslationY,
  getMatrixRotationZ,
} from "../../../../../../renderer/src/ECS/Components/Transform.ts";
import { getGameComponents } from "../../createGameWorld.ts";

const BASE_TREAD_MARK_CHANCE = 0.005;
const ANGVEL_MULTIPLIER = 0.025;
const MIN_SPEED = 0.5;
const MIN_ANGVEL = 0.1;

const treadMarkOptions: TreadMarkOptions = {
  x: 0,
  y: 0,
  width: 3,
  height: 5,
  rotation: 0,
};

export function createSpawnTreadMarksSystem({ world } = GameDI) {
  const { VehiclePartCaterpillar, RigidBodyState, CompoundPart } = getGameComponents(world);

  return (_delta: number) => {
    const caterpillarEids = query(world, [VehiclePartCaterpillar]);

    for (const eid of caterpillarEids) {
      const velEid = CompoundPart.ownerEid.get(eid);
      const linvel = RigidBodyState.linvel.getBatch(velEid);
      const angvel = RigidBodyState.angvel[velEid];

      const speed = Math.sqrt(linvel[0] * linvel[0] + linvel[1] * linvel[1]);

      if (speed < MIN_SPEED && abs(angvel) < MIN_ANGVEL) {
        continue;
      }

      const turnBonus = abs(angvel) * ANGVEL_MULTIPLIER;
      const treadMarkChance = BASE_TREAD_MARK_CHANCE + turnBonus;

      if (random() > treadMarkChance) continue;

      const globalMatrix = GlobalTransform.matrix.getBatch(eid);
      const x = getMatrixTranslationX(globalMatrix);
      const y = getMatrixTranslationY(globalMatrix);
      const rotation = getMatrixRotationZ(globalMatrix);

      treadMarkOptions.x = x;
      treadMarkOptions.y = y;
      treadMarkOptions.rotation = rotation;
      spawnTreadMark(treadMarkOptions);
    }
  };
}
