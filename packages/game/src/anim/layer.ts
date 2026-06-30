import type { EngineWorld } from "../../../engine/src/ECS/createEngineWorld.js";
import type { EntityAnimations, EntityInstance } from "../Entities/registry.js";
import { makeBlendLayer } from "./blend.js";
import { buildClipPlayer } from "./registry.js";
import type { Clip } from "./clip.js";

// A Layer is one weighted, self-clocked animation channel: `(delta, target)` eases its weight toward
// `target` (1 = active, 0 = off, skipped while ~0) and, when live, poses bones for its phase. It is
// the single animation primitive — clips and procedural stances are just two ways to make one.
export type Layer = (delta: number, target: number) => void;

// Procedural layer: hand-write `apply(phase, weight)` to pose bones directly (read `weight` to blend
// onto the current pose). For one-off, parametric, or maths-driven motion (bob, lunge reach, draw).
export function proceduralLayer(
  duration: number,
  apply: (phase: number, weight: number) => void,
  easeRate?: number,
): Layer {
  return makeBlendLayer(duration, apply, easeRate);
}

// Clip layer: data-driven from an authored `Clip`. Same shape as a procedural layer — it just gets
// its `apply` from the clip player (keyframed TRS per bone, weight-blended onto the current pose).
export function clipLayer(
  world: EngineWorld,
  clip: Clip,
  instance: Pick<EntityInstance, "root" | "bones">,
  easeRate?: number,
): Layer {
  return makeBlendLayer(clip.duration, buildClipPlayer(world, clip, instance), easeRate);
}

// One frame of an animation.
export type Anim = (delta: number) => void;

// Build one animation by hand from steps, in order. A step is either a plain Anim (e.g. a base pose)
// or a `[layer, weight]` pair driving that layer toward `weight` (1 = active, 0 = ease it out). List
// every layer the animation touches — including the ones it wants OFF — so they blend out smoothly.
export type LayerStep = Anim | [Layer, number];

export function combineLayers(...steps: LayerStep[]): Anim {
  return (delta: number) => {
    for (let i = 0; i < steps.length; i++) {
      const step = steps[i];
      if (typeof step === "function") step(delta);
      else step[0](delta, step[1]);
    }
  };
}

// The common case, wired automatically: attach extra `layers` to a unit's base animations. Each base
// animation keeps its pose but winds every extra layer down to 0 (so a layer eases out smoothly when
// you leave it); and each extra layer gets its own animation playing it at 1 over the `idle` pose,
// with the other extras held at 0. For anything non-standard (a layer over `movement`, two layers at
// once) compose it by hand with `combineLayers` instead.
export function combineAnimations(
  base: EntityAnimations,
  layers: Record<string, Layer>,
): EntityAnimations {
  const entries = Object.entries(layers);
  const out: EntityAnimations = {};

  for (const name in base) {
    out[name] = combineLayers(base[name], ...entries.map(([, layer]): LayerStep => [layer, 0]));
  }
  for (const [active] of entries) {
    out[active] = combineLayers(
      base.idle,
      ...entries.map(([name, layer]): LayerStep => [layer, name === active ? 1 : 0]),
    );
  }

  return out;
}
