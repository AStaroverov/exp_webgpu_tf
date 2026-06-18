import { addEntity, EntityId } from "bitecs";
import { randomRangeFloat, randomRangeInt } from "../../../../../../../../lib/random.ts";
import { addTransformComponents } from "renderer/src/ECS/Components/Transform.ts";
import { GameDI } from "../../../../DI/GameDI.ts";
import { MapDI } from "../../../../DI/MapDI.ts";
import { OccupantKind } from "../../../../Map/HexGrid.ts";
import { HexGridConfig } from "../../../../Map/HexConfig.ts";
import { RockConfig } from "../../../../Config/index.ts";
import { getGameComponents } from "../../../createGameWorld.ts";
import { createRockParts, getRandomStoneColor } from "./RockParts.ts";
import { generateGridRockShape } from "./generateGridRockShape.ts";
import { ObstaclePlan } from "../types.ts";

/** Inradius of a hex (center → edge midpoint) = circumradius * √3/2. */
const HEX_INRADIUS = HexGridConfig.radius * (Math.sqrt(3) / 2);
/**
 * Radius the rock fills inside the hex. Kept a bit under the inradius so the
 * stone never pokes past the cell edges; passed to the generator, which sizes
 * the rock to it directly.
 */
const ROCK_RADIUS = HEX_INRADIUS * 0.85;

/**
 * Create a rock obstacle: one boulder per footprint hex, generated to fit the
 * hex's inscribed circle. Occupies every footprint cell on the grid and records
 * them on `ObstacleFootprint`.
 */
export function createRock(
  plan: ObstaclePlan,
  { world }: Pick<typeof GameDI, "world"> = GameDI,
): EntityId | undefined {
  const grid = MapDI.grid;
  const { Obstacle, Children, ObstacleFootprint } = getGameComponents(world);

  const color = getRandomStoneColor();
  const density = RockConfig.defaultDensity;

  let rockEid: EntityId | undefined;

  for (const cell of plan.cells) {
    const center = grid.hexToWorld(cell.q, cell.r);
    if (!center) continue;

    const parts = generateGridRockShape({
      radius: ROCK_RADIUS,
      cellSize: randomRangeInt(...RockConfig.cellSizeRange),
      noiseScale: randomRangeFloat(...RockConfig.noiseScaleRange),
      noiseOctaves: randomRangeInt(...RockConfig.noiseOctavesRange),
      emptyThreshold: randomRangeFloat(...RockConfig.emptyThresholdRange),
    });
    if (parts.length === 0) continue;

    // Create the container lazily on the first non-empty hex.
    if (rockEid === undefined) {
      rockEid = addEntity(world);
      addTransformComponents(world, rockEid);
      Obstacle.addComponent(world, rockEid);
      Children.addComponent(world, rockEid);
      ObstacleFootprint.addComponent(world, rockEid);
    }

    createRockParts(rockEid, center.x, center.y, parts, color, {
      rotation: randomRangeFloat(0, Math.PI * 2),
      density,
    });

    grid.occupy(cell.q, cell.r, rockEid, OccupantKind.Obstacle);
    ObstacleFootprint.add(rockEid, cell.q, cell.r);
  }

  return rockEid;
}
