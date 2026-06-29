import type { EngineWorld } from "../../../../engine/src/ECS/createEngineWorld.ts";
import { buildTree } from "./tree.ts";
import { buildUnit } from "./unit.ts";
import { buildLightsaber } from "./lightsaber.ts";
import { buildSwordsman } from "./swordsman.ts";

export type EntityAnimations = Record<string, (delta: number) => void>;

export type EntityOptions = { scale: number };

export type EntityInstance = {
  root: number;
  bones: Record<string, number>;
  animations: EntityAnimations;
};

export type EntityDef = {
  id: string;
  label: string;
  build: (world: EngineWorld, options: EntityOptions) => EntityInstance;
};

export const ENTITIES: EntityDef[] = [
  { id: "tree", label: "Tree", build: buildTree },
  { id: "unit", label: "Unit", build: buildUnit },
  { id: "sword", label: "Lightsaber", build: buildLightsaber },
  {
    id: "swordsman",
    label: "Swordsman",
    build: (world, options) =>
      buildSwordsman(world, { ...options, parts: { unit: buildUnit, sword: buildLightsaber } }),
  },
];
