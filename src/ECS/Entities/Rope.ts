import { addComponent, addEntity, World } from 'bitecs';
import { addTransformComponents, applyMatrixTranslate, LocalTransform } from '../Components/Transform.ts';
import { Color, setColor, Thinness } from '../Components/Common.ts';
import { Rope } from '../Components/Rope.ts';

export function createRope(world: World, { x, y, thinness, color, points }: {
    x: number;
    y: number;
    thinness: number;
    color: [number, number, number, number];
    points: number[] | Float32Array;
}) {
    const id = addEntity(world);
    addComponent(world, id, LocalTransform);
    addComponent(world, id, Thinness);
    addComponent(world, id, Color);
    addComponent(world, id, Rope);

    addTransformComponents(world, id);
    applyMatrixTranslate(LocalTransform.matrix.getBatche(id), x, y);

    Rope.points.setBatch(id, points);
    Thinness.value[id] = thinness;
    setColor(id, color[0], color[1], color[2], color[3]);

    return id;
}