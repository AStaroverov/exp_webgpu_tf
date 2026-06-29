import { mat4 } from "gl-matrix";
import {
  getEngineComponents,
  type EngineWorld,
} from "../../../../engine/src/ECS/createEngineWorld.ts";
import { makeBlendLayer } from "../anim/blend.ts";
import { buildClipPlayer } from "../anim/registry.ts";
import { SWORD_SWING } from "../anim/presets/index.ts";
import type { EntityAnimations, EntityInstance, EntityOptions } from "./registry.ts";
import type { UnitInstance } from "./unit.ts";

export type SwordsmanParts = {
  unit: (world: EngineWorld, options: EntityOptions) => UnitInstance;
  sword: (world: EngineWorld, options: EntityOptions) => EntityInstance;
};

type Key = { at: number; v: number };

const DEG = Math.PI / 180;
const SWORD_REL = 1.5;

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
// is parented to the hand, so it follows the arm in every animation. `sword_slice` plays an authored
// clip (anim/presets) that keys the arm; `lunge` is a procedural stance whose weight eases in while
// active and out otherwise, so the rest ↔ lunge transition blends smoothly.
export function buildSwordsman(
  world: EngineWorld,
  { scale, parts }: EntityOptions & { parts: SwordsmanParts },
): EntityInstance {
  const { Children, LocalTransform } = getEngineComponents(world);

  const unit = parts.unit(world, { scale });
  const sword = parts.sword(world, { scale: SWORD_REL });
  Children.addChild(unit.bones.armR, sword.root);

  const bones = { ...prefix("unit/", unit.bones), ...prefix("sword/", sword.bones) };

  const rootMatrix = LocalTransform.matrix.getBatch(unit.root);
  const armMatrix = LocalTransform.matrix.getBatch(unit.bones.armR);
  const restArmX = armMatrix[12];
  const restArmZ = armMatrix[14];

  const lunge = makeBlendLayer(LUNGE_CYCLE, (phase, weight) => {
    const reach = sampleKeys(LUNGE_KEYS, phase);
    let px = restArmX;
    let py = 0;
    let pz = restArmZ;
    let rx = 0;
    px += (LUNGE_HAND[0] - px) * weight;
    py += (LUNGE_HAND[1] + reach * LUNGE_REACH - py) * weight;
    pz += (LUNGE_HAND[2] - pz) * weight;
    rx += (LUNGE_ROT_X - rx) * weight;

    mat4.identity(armMatrix);
    mat4.translate(armMatrix, armMatrix, [px, py, pz]);
    mat4.rotateX(armMatrix, armMatrix, rx);

    rootMatrix[13] += Math.max(0, reach) * LUNGE_BODY * weight;
  });

  const slice = makeBlendLayer(
    SWORD_SWING.duration,
    buildClipPlayer(world, SWORD_SWING, { root: unit.root, bones }),
  );

  const animations: EntityAnimations = {};
  for (const name in unit.animations) {
    const unitAnim = unit.animations[name];
    animations[name] = (delta: number) => {
      unitAnim(delta);
      lunge(delta, 0);
      slice(delta, 0);
    };
  }
  animations.sword_slice = (delta: number) => {
    unit.animations.idle?.(delta);
    lunge(delta, 0);
    slice(delta, 1);
  };
  animations.lunge = (delta: number) => {
    unit.animations.idle?.(delta);
    lunge(delta, 1);
    slice(delta, 0);
  };

  return { root: unit.root, bones, animations };
}

function prefix(p: string, m: Record<string, number>): Record<string, number> {
  return Object.fromEntries(Object.entries(m).map(([k, v]) => [p + k, v]));
}
