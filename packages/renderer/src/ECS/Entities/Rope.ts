import { addEntity } from "bitecs";
import { addTransformComponents, applyMatrixTranslate } from "../Components/Transform.ts";
import { getRenderComponents, type RenderWorldLike } from "../world.ts";

export function createRope(
  world: RenderWorldLike,
  {
    x,
    y,
    thinness,
    color,
    points,
  }: {
    x: number;
    y: number;
    thinness: number;
    color: [number, number, number, number];
    points: number[] | Float32Array;
  },
) {
  const { Color, LocalTransform, Rope, Thinness } = getRenderComponents(world);
  const id = addEntity(world);

  addTransformComponents(world, id);
  applyMatrixTranslate(LocalTransform.matrix.getBatch(id), x, y, 0);

  Rope.addComponent(world, id, points);
  Color.addComponent(world, id, color[0], color[1], color[2], color[3]);
  Thinness.addComponent(world, id, thinness);

  return id;
}
