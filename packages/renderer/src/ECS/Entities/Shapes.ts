import { addEntity, World } from 'bitecs';
import { addTransformComponents, applyMatrixTranslate, LocalTransform } from '../Components/Transform.ts';
import { Color, Roundness, TColor } from '../Components/Common.ts';
import { Shape, ShapeKind } from '../Components/Shape.ts';

export function createCircle(world: World, { x, y, z, radius, color }: {
    x: number;
    y: number;
    z: number;
    radius: number;
    color: TColor;
}) {
    const id = addEntity(world);

    addTransformComponents(world, id);
    applyMatrixTranslate(LocalTransform.matrix.getBatch(id), x, y, z);

    Shape.addComponent(world, id, ShapeKind.Circle, radius);
    Color.addComponent(world, id, color[0], color[1], color[2], color[3]);

    return id;
}

export function createRectangle(world: World, { x, y, z, width, height, color, roundness }: {
    x: number;
    y: number;
    z: number;
    width: number;
    height: number;
    color: TColor;
    roundness?: number;
}) {
    const id = addEntity(world);

    addTransformComponents(world, id);
    applyMatrixTranslate(LocalTransform.matrix.getBatch(id), x, y, z);

    Shape.addComponent(world, id, ShapeKind.Rectangle, width, height);
    Color.addComponent(world, id, color[0], color[1], color[2], color[3]);
    Roundness.addComponent(world, id, roundness ?? 0);

    return id;
}

export function createTriangle(world: World, { x, y, z, color, roundness, point1, point2, point3 }: {
    x: number;
    y: number;
    z: number;
    roundness?: number;
    point1: [number, number];
    point2: [number, number];
    point3: [number, number];
    color: TColor;
}) {
    const id = addEntity(world);

    addTransformComponents(world, id);
    applyMatrixTranslate(LocalTransform.matrix.getBatch(id), x, y, z);

    Shape.addComponent(world, id, ShapeKind.Triangle, point1[0], point1[1], point2[0], point2[1], point3[0], point3[1]);
    Color.addComponent(world, id, color[0], color[1], color[2], color[3]);
    Roundness.addComponent(world, id, roundness ?? 0);

    return id;
}