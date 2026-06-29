import { mat4 } from "gl-matrix";
import {
  getEngineComponents,
  type EngineWorld,
} from "../../../../engine/src/ECS/createEngineWorld.ts";
import { combineAnimations, proceduralLayer } from "../anim/layer.ts";
import { sampleKeys, type Key } from "../anim/curve.ts";
import type { EntityInstance, EntityOptions } from "./registry.ts";
import type { UnitInstance } from "./unit.ts";
import {
  applyMatrixRotateZ,
  applyMatrixTranslate,
} from "../../../../renderer/src/ECS/Components/Transform.ts";

export type ArcherParts = {
  unit: (world: EngineWorld, options: EntityOptions) => UnitInstance;
  bow: (world: EngineWorld, options: EntityOptions) => EntityInstance;
};

const DEG = Math.PI / 180;
const BOW_REL = 1.3;
const BOW_HALF_HEIGHT = 1.0; // bow grip sits at z = HALF_HEIGHT in bow space; center it on the hand.

// Bow hand (left) holds the bow extended forward and steady; draw hand (right) pulls the nocked
// arrow back to the anchor, holds, then releases forward.
const BOW_HAND: [number, number, number] = [-0.4, 1.4, 2.5];
const BOW_DROP = 1.0; // bow hand starts this much lower and rises proportionally with the draw.
const BOW_TILT = 35 * DEG; // bow hand tilts back (relative to the body) as it rises with the draw.
const DRAW_HAND: [number, number, number] = [0.4, 1.1, 1.5]; // held at ~rest height; the draw only pulls back (-Y).
const DRAW_REACH = 0.75; // how far the draw hand pulls back (-Y) at full draw.
const LEAN_BACK = 18 * DEG; // body + head tilt back about the hips at full draw.
const SHOOT_CYCLE = 1.8;
// pull = 1 at full draw (hand back at the anchor), pull < 0 is the release follow-through forward.
const PULL_KEYS: Key[] = [
  { at: 0.0, v: 0 },
  { at: 0.52, v: 1 }, // slow build to full draw
  { at: 0.62, v: 1 }, // hold at the anchor
  { at: 0.68, v: 0.7 }, // fast release
  { at: 1.0, v: 0 },
];

const tiltScratch = mat4.create();

// An archer = a unit holding a bow in its LEFT hand and a nocked arrow in its RIGHT (draw) hand.
// The bow is centered on the hand and stays vertical; the arrow is laid forward (+Y) and parented to
// the draw hand so it pulls back with it. `shoot` is a procedural cycle whose weight eases in while
// active, so rest ↔ draw blends smoothly (same shape as the swordsman's stances).
export function buildArcher(
  world: EngineWorld,
  { scale, parts }: EntityOptions & { parts: ArcherParts },
): EntityInstance {
  const { Children, LocalTransform } = getEngineComponents(world);

  const unit = parts.unit(world, { scale });
  const bow = parts.bow(world, { scale: BOW_REL });

  // Bow: center its grip on the left hand (slide down by the grip height).
  Children.addChild(unit.bones.armL, bow.root);
  applyMatrixTranslate(LocalTransform.matrix.getBatch(bow.root), 0, -0.3, -BOW_HALF_HEIGHT);
  applyMatrixRotateZ(LocalTransform.matrix.getBatch(bow.root), (180 * Math.PI) / 180);

  const bones = {
    ...prefix("unit/", unit.bones),
    ...prefix("bow/", bow.bones),
  };

  const bodyMatrix = LocalTransform.matrix.getBatch(unit.bones.body);
  const headMatrix = LocalTransform.matrix.getBatch(unit.bones.head);
  const armLMatrix = LocalTransform.matrix.getBatch(unit.bones.armL);
  const armRMatrix = LocalTransform.matrix.getBatch(unit.bones.armR);
  const restArmLX = armLMatrix[12];
  const restArmLZ = armLMatrix[14];
  const restArmRX = armRMatrix[12];
  const restArmRZ = armRMatrix[14];

  function poseArm(
    m: mat4,
    restX: number,
    restZ: number,
    target: [number, number, number],
    rotX: number,
    weight: number,
  ): void {
    const px = restX + (target[0] - restX) * weight;
    const py = target[1] * weight;
    const pz = restZ + (target[2] - restZ) * weight;
    mat4.identity(m);
    mat4.translate(m, m, [px, py, pz]);
    mat4.rotateX(m, m, rotX * weight);
  }

  const shoot = proceduralLayer(SHOOT_CYCLE, (phase, weight) => {
    const pull = sampleKeys(PULL_KEYS, phase);
    const drawn = Math.max(0, pull);
    poseArm(
      armLMatrix,
      restArmLX,
      restArmLZ,
      [BOW_HAND[0], BOW_HAND[1], BOW_HAND[2] - (1 - drawn) * BOW_DROP],
      BOW_TILT * drawn,
      weight,
    );
    poseArm(
      armRMatrix,
      restArmRX,
      restArmRZ,
      [DRAW_HAND[0], DRAW_HAND[1] - pull * DRAW_REACH, DRAW_HAND[2]],
      BOW_TILT * drawn,
      weight,
    );
    // Lean body + head back about the hips (pre-multiply a root-space X rotation onto each bone).
    mat4.fromXRotation(tiltScratch, LEAN_BACK * drawn * weight);
    mat4.multiply(bodyMatrix, tiltScratch, bodyMatrix);
    mat4.multiply(headMatrix, tiltScratch, headMatrix);
  });

  const animations = combineAnimations(unit.animations, { shoot });

  return { root: unit.root, bones, animations };
}

function prefix(p: string, m: Record<string, number>): Record<string, number> {
  return Object.fromEntries(Object.entries(m).map(([k, v]) => [p + k, v]));
}
