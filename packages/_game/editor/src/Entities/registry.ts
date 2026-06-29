import type { EngineWorld } from "../../../../engine/src/ECS/createEngineWorld.ts";
import { buildTree } from "./tree.ts";
import { buildUnit } from "./unit.ts";

export type EntityDef = {
  id: string;
  label: string;
  build: (world: EngineWorld) => number;
};

export const ENTITIES: EntityDef[] = [
  { id: "tree", label: "Tree", build: buildTree },
  { id: "unit", label: "Unit", build: buildUnit },
];
