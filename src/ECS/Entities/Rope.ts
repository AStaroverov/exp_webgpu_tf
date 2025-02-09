import { addComponent, addEntity, IWorld } from 'bitecs';
import { addTransformComponents, applyMatrixTranslate, LocalTransform } from '../Components/Transform.ts';
import { Color, setColor, Thinness } from '../Components/Common.ts';
import { Rope } from '../Components/Rope.ts';

export function createRope(world: IWorld, { x, y, thinness, color, points }: {
    x: number;
    y: number;
    thinness: number;
    color: [number, number, number, number];
    points: number[] | Float32Array;
}) {
    const id = addEntity(world);
    addComponent(world, LocalTransform, id);
    addComponent(world, Thinness, id);
    addComponent(world, Color, id);
    addComponent(world, Rope, id);

    addTransformComponents(world, id);
    applyMatrixTranslate(LocalTransform.matrix[id], x, y);

    Rope.points[id].set(points);
    Thinness.value[id] = thinness;
    setColor(id, color[0], color[1], color[2], color[3]);

    return id;
}