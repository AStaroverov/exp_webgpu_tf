import { mat4 } from "gl-matrix";
import { createRectangle, createTrapezoid } from "../../../../renderer/src/ECS/Entities/Shapes.ts";
import {
  addTransformComponents,
  applyMatrixScale,
  applyMatrixTranslate,
} from "../../../../renderer/src/ECS/Components/Transform.ts";
import type { TColor } from "../../../../renderer/src/ECS/Components/Common.ts";
import {
  createEntityId,
  getEngineComponents,
  type EngineWorld,
} from "../../../../engine/src/ECS/createEngineWorld.ts";
import { proceduralLayer, type Layer } from "../anim/layer.ts";
import type { EntityAnimations, EntityInstance, EntityOptions } from "./registry.ts";

const COLOR: TColor = [0.13, 0.34, 0.56, 1];
const HOVER = 0.7;

// A unit exposes its right hand as a weapon attach point (for composing, e.g. a swordsman).
export type UnitInstance = EntityInstance & { hand: number };

export function buildUnit(world: EngineWorld, { scale }: EntityOptions): UnitInstance {
  const parts = buildStructure(world, scale);
  const bones = {
    root: parts.root,
    body: parts.body,
    armL: parts.armL,
    armR: parts.armR,
    head: parts.head,
  };
  return { root: parts.root, bones, animations: buildAnimations(world, parts), hand: parts.armR };
}

type UnitParts = {
  root: number;
  body: number;
  armL: number;
  armR: number;
  head: number;
  bodyHeight: number;
  armX: number;
  armZ: number;
  headZ: number;
  scale: number;
};

function buildStructure(world: EngineWorld, scale: number): UnitParts {
  const { Children, LocalTransform } = getEngineComponents(world);

  const root = createEntityId(world);
  addTransformComponents(world, root);
  const rootMatrix = LocalTransform.matrix.getBatch(root);
  applyMatrixTranslate(rootMatrix, 0, 0, HOVER);
  applyMatrixScale(rootMatrix, scale, scale, scale);
  Children.addComponent(world, root);

  const add = (eid: number) => Children.addChild(root, eid);

  const bodyHeight = 2.2;
  const body = createTrapezoid(world, {
    x: 0,
    y: 0,
    z: bodyHeight / 2,
    topWidth: 1.4,
    bottomWidth: 0.4,
    height: bodyHeight,
    depth: 1.0,
    color: COLOR,
    eid: createEntityId(world),
  });
  mat4.rotateX(
    LocalTransform.matrix.getBatch(body),
    LocalTransform.matrix.getBatch(body),
    Math.PI / 2,
  );
  add(body);

  const armSize = 0.35;
  const armZ = bodyHeight * 0.45;
  const armX = 0.9;
  const armL = createRectangle(world, {
    x: -armX,
    y: 0,
    z: armZ,
    width: armSize,
    height: armSize,
    depth: armSize,
    color: COLOR,
    eid: createEntityId(world),
  });
  add(armL);
  Children.addComponent(world, armL);
  const armR = createRectangle(world, {
    x: armX,
    y: 0,
    z: armZ,
    width: armSize,
    height: armSize,
    depth: armSize,
    color: COLOR,
    eid: createEntityId(world),
  });
  add(armR);
  Children.addComponent(world, armR);

  const headSize = 0.6;
  const headGap = 0.3;
  const headZ = bodyHeight + headGap + headSize / 2;
  const head = createRectangle(world, {
    x: 0,
    y: 0,
    z: headZ,
    width: headSize,
    height: headSize,
    depth: headSize,
    color: COLOR,
    eid: createEntityId(world),
  });
  add(head);

  return { root, body, armL, armR, head, bodyHeight, armX, armZ, headZ, scale };
}

type Pose = {
  headTilt: number;
  height: number;
  bobW: number;
  roll: number;
};

function buildAnimations(world: EngineWorld, p: UnitParts): EntityAnimations {
  const { LocalTransform } = getEngineComponents(world);
  const { root, body, armL, armR, head, bodyHeight, armX, armZ, headZ, scale } = p;

  const REST: Pose = { headTilt: 0, height: HOVER, bobW: 1, roll: 0 };
  const MOVE: Pose = { headTilt: 0.25, height: HOVER, bobW: 1, roll: -0.25 };
  const DEAD: Pose = { headTilt: 0, height: 0, bobW: 0, roll: 1.5 };
  const poseKeys = Object.keys(REST) as (keyof Pose)[];
  const pose: Pose = { ...REST };
  let clock = 0;

  // movement/death are weighted layers added onto the rest pose (REST + Σ weight·(target − REST)),
  // so each eases in/out on its own and is skipped while its weight is ~0 — same as the swordsman stances.
  function overlay(target: Pose): Layer {
    return proceduralLayer(
      1,
      (_phase, weight) => {
        for (const k of poseKeys) pose[k] += (target[k] - REST[k]) * weight;
      },
      6,
    );
  }
  const move = overlay(MOVE);
  const death = overlay(DEAD);

  function applyPose(): void {
    const rm = LocalTransform.matrix.getBatch(root);
    mat4.identity(rm);
    mat4.translate(rm, rm, [0, 0, pose.height + 0.25 * pose.bobW * Math.sin(clock * 2)]);
    mat4.rotateX(rm, rm, pose.roll);
    mat4.scale(rm, rm, [scale, scale, scale]);

    const bm = LocalTransform.matrix.getBatch(body);
    mat4.identity(bm);
    mat4.translate(bm, bm, [0, 0, bodyHeight / 2]);
    mat4.rotateX(bm, bm, Math.PI / 2);

    const hm = LocalTransform.matrix.getBatch(head);
    mat4.identity(hm);
    mat4.translate(hm, hm, [0, 0, headZ]);
    mat4.rotateX(hm, hm, pose.headTilt);

    for (const [eid, x] of [
      [armL, -armX],
      [armR, armX],
    ] as const) {
      const am = LocalTransform.matrix.getBatch(eid);
      mat4.identity(am);
      mat4.translate(am, am, [x, 0, armZ]);
    }
  }

  function frame(delta: number, moveTarget: number, deathTarget: number): void {
    clock += delta;
    Object.assign(pose, REST);
    move(delta, moveTarget);
    death(delta, deathTarget);
    applyPose();
  }

  return {
    idle: (delta) => frame(delta, 0, 0),
    death: (delta) => frame(delta, 0, 1),
    movement: (delta) => frame(delta, 1, 0),
  };
}
