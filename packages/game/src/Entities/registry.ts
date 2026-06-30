import type { EngineWorld } from "../../../engine/src/ECS/createEngineWorld.js";
import { buildTree } from "./tree.js";
import { buildUnit } from "./unit.js";
import { buildLightsaber } from "./lightsaber.js";
import { buildBow } from "./bow.js";
import { buildArrow } from "./arrow.js";
import { buildSwordsman } from "./swordsman.js";
import { buildArcher } from "./archer.js";

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
  { id: "bow", label: "Bow", build: buildBow },
  { id: "arrow", label: "Arrow", build: buildArrow },
  {
    id: "swordsman",
    label: "Swordsman",
    build: (world, options) =>
      buildSwordsman(world, { ...options, parts: { unit: buildUnit, sword: buildLightsaber } }),
  },
  {
    id: "archer",
    label: "Archer",
    build: (world, options) =>
      buildArcher(world, {
        ...options,
        parts: { unit: buildUnit, bow: buildBow },
      }),
  },
];
