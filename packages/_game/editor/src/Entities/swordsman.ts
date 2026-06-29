import { mat4 } from "gl-matrix";
import {
  getEngineComponents,
  type EngineWorld,
} from "../../../../engine/src/ECS/createEngineWorld.ts";
import type { EntityAnimations, EntityInstance, EntityOptions } from "./registry.ts";
import type { UnitInstance } from "./unit.ts";

export type SwordsmanParts = {
  unit: (world: EngineWorld, options: EntityOptions) => UnitInstance;
  sword: (world: EngineWorld, options: EntityOptions) => EntityInstance;
};

type Key = { at: number; v: number };

const DEG = Math.PI / 180;
const SWORD_REL = 1.5;
const BODY_YAW = 35 * DEG;

// Slice = an overhead chop. Wind-up (0 → ~0.6): the hand rises to head level and out to the right,
// blade pointing up. Strike (~0.6 → 1): the hand sweeps down to the body centre, the blade pitches
// forward/down, the wrist rolls through, and the body steps in. It ends at the struck pose (= the
// cycle start), so there is no orphan recovery phase. Channels share one phase.
const SWING_CYCLE = 1.4;
const SLICE_X_KEYS: Key[] = [
  { at: 0, v: 0 },
  { at: 0.6, v: 0.95 },
  { at: 1, v: 0 },
];
const SLICE_Y_KEYS: Key[] = [
  { at: 0, v: 0.6 },
  { at: 0.6, v: -0.3 },
  { at: 1, v: 0.6 },
];
const SLICE_Z_KEYS: Key[] = [
  { at: 0, v: 1.0 },
  { at: 0.6, v: 2.4 },
  { at: 1, v: 1.0 },
];
const SLICE_PITCH_KEYS: Key[] = [
  { at: 0, v: -100 * DEG },
  { at: 0.6, v: 60 * DEG },
  { at: 0.8, v: -110 * DEG },
  { at: 1, v: -100 * DEG },
];
const SLICE_WRIST_KEYS: Key[] = [
  { at: 0, v: 0 },
  { at: 0.6, v: -0.9 },
  { at: 0.8, v: 0.9 },
  { at: 1, v: 0 },
];
const SLICE_STEP_KEYS: Key[] = [
  { at: 0, v: 0 },
  { at: 0.6, v: 0 },
  { at: 0.8, v: 1 },
  { at: 1, v: 0 },
];
const SLICE_STEP = 0.5;

// Lunge stance: arm extended forward, blade horizontal (a thrust); the body steps forward.
const LUNGE_HAND: [number, number, number] = [0.9, 0.6, 1.0];
const LUNGE_ROT_X = -90 * DEG;
const LUNGE_REACH = 0.7;
const LUNGE_BODY = 0.9;
const LUNGE_CYCLE = 1.4;
// reach < 0 cocks the arm back behind the body (wind-up); reach > 0 extends it forward (thrust).
const LUNGE_KEYS: Key[] = [
  { at: 0.0, v: 0 },
  { at: 0.5, v: -1 },
  { at: 0.62, v: 1 },
  { at: 1.0, v: 0 },
];

function smoothstep(x: number): number {
  return x * x * (3 - 2 * x);
}

function sampleKeys(keys: Key[], p: number): number {
  for (let i = 0; i < keys.length - 1; i++) {
    const a = keys[i];
    const b = keys[i + 1];
    if (p <= b.at) return a.v + (b.v - a.v) * smoothstep((p - a.at) / (b.at - a.at));
  }
  return keys[keys.length - 1].v;
}

// A swordsman = a unit holding a weapon in its right hand (the unit's exposed `hand`). The weapon
// is parented to the hand, so it follows the arm in every animation. Combat stances (slice, lunge)
// each have an independent weight that eases in while their animation is active and out otherwise,
// so transitions between rest and either stance — and between the stances — blend smoothly.
export function buildSwordsman(
  world: EngineWorld,
  { scale, parts }: EntityOptions & { parts: SwordsmanParts },
): EntityInstance {
  const { Children, LocalTransform } = getEngineComponents(world);

  const unit = parts.unit(world, { scale });
  const sword = parts.sword(world, { scale: SWORD_REL });
  Children.addChild(unit.hand, sword.root);

  const rootMatrix = LocalTransform.matrix.getBatch(unit.root);
  const armMatrix = LocalTransform.matrix.getBatch(unit.hand);
  const restArmX = armMatrix[12];
  const restArmZ = armMatrix[14];

  let t = 0;
  let sliceW = 0;
  let lungeW = 0;
  function applyStance(delta: number, sliceTarget: number, lungeTarget: number): void {
    const ease = 1 - Math.exp(-delta * 8);
    sliceW += (sliceTarget - sliceW) * ease;
    lungeW += (lungeTarget - lungeW) * ease;
    t += delta;

    const sp = (t % SWING_CYCLE) / SWING_CYCLE;
    const lp = (t % LUNGE_CYCLE) / LUNGE_CYCLE;
    const reach = sampleKeys(LUNGE_KEYS, lp);

    // Blend arm pose params from rest → slice (sliceW) → lunge (lungeW).
    let px = restArmX;
    let py = 0;
    let pz = restArmZ;
    let rx = 0;
    let ry = 0;
    px += (sampleKeys(SLICE_X_KEYS, sp) - px) * sliceW;
    py += (sampleKeys(SLICE_Y_KEYS, sp) - py) * sliceW;
    pz += (sampleKeys(SLICE_Z_KEYS, sp) - pz) * sliceW;
    rx += (sampleKeys(SLICE_PITCH_KEYS, sp) - rx) * sliceW;
    ry += (sampleKeys(SLICE_WRIST_KEYS, sp) - ry) * sliceW;
    px += (LUNGE_HAND[0] - px) * lungeW;
    py += (LUNGE_HAND[1] + reach * LUNGE_REACH - py) * lungeW;
    pz += (LUNGE_HAND[2] - pz) * lungeW;
    rx += (LUNGE_ROT_X - rx) * lungeW;
    ry += (0 - ry) * lungeW;

    mat4.identity(armMatrix);
    mat4.translate(armMatrix, armMatrix, [px, py, pz]);
    mat4.rotateX(armMatrix, armMatrix, rx);
    mat4.rotateY(armMatrix, armMatrix, ry);

    mat4.rotateZ(rootMatrix, rootMatrix, BODY_YAW * sliceW);
    rootMatrix[13] +=
      sampleKeys(SLICE_STEP_KEYS, sp) * SLICE_STEP * sliceW +
      Math.max(0, reach) * LUNGE_BODY * lungeW;
  }

  const animations: EntityAnimations = {};
  for (const name in unit.animations) {
    const unitAnim = unit.animations[name];
    animations[name] = (delta: number) => {
      unitAnim(delta);
      applyStance(delta, 0, 0);
    };
  }
  animations.sword_slice = (delta: number) => {
    unit.animations.idle?.(delta / 10);
    applyStance(delta / 10, 1, 0);
  };
  animations.lunge = (delta: number) => {
    unit.animations.idle?.(delta);
    applyStance(delta, 0, 1);
  };

  return { root: unit.root, animations };
}
