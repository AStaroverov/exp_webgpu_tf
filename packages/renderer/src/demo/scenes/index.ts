import type { RenderWorld } from "../../ECS/world.ts";
import type { DemoScene } from "./types.ts";
import { createEmitterScene } from "./emitter.ts";
import { createShowcaseScene } from "./showcase.ts";
import { createFinalScene } from "./final.ts";
import { createPerfScene } from "./perf.ts";
import { createPerf2Scene } from "./perf2.ts";
import { createSwarmScene } from "./swarm.ts";

// Scene registry (picked live from the GUI; persisted in localStorage, applied on reload):
//   "emitter"  — ground + box occluder + a GUI-movable/resizable emitter sphere.
//   "showcase" — one of every shape kind + several lights.
//   "final" / "perf" / "perf2" — animated final scene / perf-cost harnesses.
//   "swarm"    — MANY small emitters hovering over the floor + sparse large occluders (the
//                many-lights harness: round-robin subsample + clustered light culling).
export const SCENE_OPTIONS = ["emitter", "showcase", "final", "perf", "perf2", "swarm"] as const;
export type SceneName = (typeof SCENE_OPTIONS)[number];

const factories: Record<SceneName, (world: RenderWorld) => DemoScene> = {
  emitter: createEmitterScene,
  showcase: createShowcaseScene,
  final: createFinalScene,
  perf: createPerfScene,
  perf2: createPerf2Scene,
  swarm: createSwarmScene,
};

// Spawns the scene's world content and returns its hooks (animate / setupGUI / cameraZoom).
export function createScene(name: SceneName, world: RenderWorld): DemoScene {
  return factories[name](world);
}

export type { DemoScene } from "./types.ts";
