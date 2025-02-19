import { addEntity, World } from 'bitecs';
import { addTransformComponents, applyMatrixTranslate, LocalTransform } from '../Components/Transform.ts';
import { ColorMethods, ThinnessMethods } from '../Components/Common.ts';
import { RopeMethods } from '../Components/Rope.ts';

export function createRope(world: World, { x, y, thinness, color, points }: {
    x: number;
    y: number;
    thinness: number;
    color: [number, number, number, number];
    points: number[] | Float32Array;
}) {
    const id = addEntity(world);

    addTransformComponents(world, id);
    applyMatrixTranslate(LocalTransform.matrix.getBatche(id), x, y);

    RopeMethods.addComponent(id, points);
    ColorMethods.addComponent(id, color[0], color[1], color[2], color[3]);
    ThinnessMethods.addComponent(id, thinness);

    return id;
}