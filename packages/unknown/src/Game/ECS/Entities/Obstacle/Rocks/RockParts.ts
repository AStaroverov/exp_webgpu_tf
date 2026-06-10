import { RigidBodyType } from "@dimforge/rapier2d-simd";
import { EntityId } from "bitecs";
import { min } from "../../../../../../../../lib/math.ts";
import { randomRangeFloat, randomRangeInt } from "../../../../../../../../lib/random.ts";
import { TColor } from "renderer/src/ECS/Components/Common.ts";
import { GameDI } from "../../../../DI/GameDI.ts";
import { CollisionGroup } from "../../../../Physical/createRigid.ts";
import { ZIndexConfig } from "../../../../Config/index.ts";
import { createRectangleRR } from "../../../Components/RigidRender.ts";
import { getGameComponents } from "../../../createGameWorld.ts";

export type RockPartData = [x: number, y: number, w: number, h: number, rotation: number];

// Stone colors — various shades of gray and brown.
export const STONE_COLORS: TColor[] = [
  new Float32Array([0.35, 0.33, 0.3, 1]),
  new Float32Array([0.42, 0.4, 0.36, 1]),
  new Float32Array([0.38, 0.35, 0.32, 1]),
  new Float32Array([0.45, 0.42, 0.38, 1]),
  new Float32Array([0.32, 0.3, 0.28, 1]),
  new Float32Array([0.4, 0.38, 0.35, 1]),
  new Float32Array([0.48, 0.45, 0.4, 1]),
  new Float32Array([0.36, 0.34, 0.3, 1]),
];

export function getRandomStoneColor(): TColor {
  return STONE_COLORS[randomRangeInt(0, STONE_COLORS.length - 1)];
}

const tmpColor = new Float32Array([0.4, 0.38, 0.35, 1]);

/**
 * Create rock parts as fixed bodies, parented to `rockEid`. Local part anchors
 * (relative to the rock origin) are rotated by `options.rotation` and offset by
 * (rockX, rockY) into world space. Ported from `packages/tanks`.
 */
export function createRockParts(
  rockEid: EntityId,
  rockX: number,
  rockY: number,
  partsData: RockPartData[],
  baseColor: TColor,
  options: {
    rotation: number;
    density: number;
  },
  { world } = GameDI,
): EntityId[] {
  const { Obstacle, Parent, Children, Hitable, Damagable } = getGameComponents(world);
  const partEids: EntityId[] = [];

  const cos = Math.cos(options.rotation);
  const sin = Math.sin(options.rotation);

  for (let i = 0; i < partsData.length; i++) {
    const [anchorX, anchorY, width, height, partRotation] = partsData[i];

    // Transform anchor from local to world space.
    const worldX = anchorX * cos - anchorY * sin;
    const worldY = anchorX * sin + anchorY * cos;

    // Slightly vary the color for each part.
    const v = randomRangeFloat(-0.05, 0.05);
    tmpColor[0] = Math.max(0, Math.min(1, baseColor[0] + v));
    tmpColor[1] = Math.max(0, Math.min(1, baseColor[1] + v));
    tmpColor[2] = Math.max(0, Math.min(1, baseColor[2] + v));
    tmpColor[3] = 1;

    const [partEid] = createRectangleRR({
      x: rockX + worldX,
      y: rockY + worldY,
      z: ZIndexConfig.Rock + (width + height) / 2,
      width,
      height,
      rotation: options.rotation + partRotation,
      color: tmpColor,
      density: options.density,
      bodyType: RigidBodyType.Fixed,
      belongsCollisionGroup: CollisionGroup.OBSTACLE,
      interactsCollisionGroup: CollisionGroup.ALL,
      belongsSolverGroup: CollisionGroup.ALL,
      interactsSolverGroup: CollisionGroup.ALL,
    });

    Obstacle.addComponent(world, partEid);
    Parent.addComponent(world, partEid, rockEid);
    Children.addChildren(rockEid, partEid);

    // Hitable/Damagable for a future destruction system (Decision 5, deferred).
    const health = min(width, height) * 2;
    Hitable.addComponent(world, partEid, health);
    Damagable.addComponent(world, partEid, min(width, height) / 10);

    partEids.push(partEid);
  }

  return partEids;
}
