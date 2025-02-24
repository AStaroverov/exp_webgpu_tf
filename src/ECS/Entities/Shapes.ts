import { addEntity, World } from 'bitecs';
import { addTransformComponents, applyMatrixTranslate, LocalTransform } from '../Components/Transform.ts';
import { Color, Roundness, Shadow, TColor, TShadow } from '../Components/Common.ts';
import { Shape, ShapeKind } from '../Components/Shape.ts';

export function createCircle(world: World, { x, y, radius, color, shadow }: {
    x: number;
    y: number;
    radius: number;
    color: TColor;
    shadow?: TShadow;
}) {
    const id = addEntity(world);

    addTransformComponents(world, id);
    applyMatrixTranslate(LocalTransform.matrix.getBatche(id), x, y);

    Shape.addComponent(world, id, ShapeKind.Circle, radius);
    Color.addComponent(world, id, color[0], color[1], color[2], color[3]);
    Shadow.addComponent(world, id, shadow?.[0] ?? 0, shadow?.[1] ?? 0);

    return id;
}

export function createRectangle(world: World, { x, y, width, height, color, roundness, shadow }: {
    x: number;
    y: number;
    width: number;
    height: number;
    color: TColor;
    roundness?: number;
    shadow?: TShadow;
}) {
    const id = addEntity(world);

    addTransformComponents(world, id);
    applyMatrixTranslate(LocalTransform.matrix.getBatche(id), x, y);

    Shape.addComponent(world, id, ShapeKind.Rectangle, width, height);
    Color.addComponent(world, id, color[0], color[1], color[2], color[3]);
    Shadow.addComponent(world, id, shadow?.[0] ?? 0, shadow?.[1] ?? 0);
    Roundness.addComponent(world, id, roundness ?? 0);

    return id;
}

export function createTriangle(world: World, { x, y, color, roundness, point1, point2, point3, shadow }: {
    x: number;
    y: number;
    roundness?: number;
    point1: [number, number];
    point2: [number, number];
    point3: [number, number];
    color: TColor;
    shadow?: TShadow;
}) {
    const id = addEntity(world);

    addTransformComponents(world, id);
    applyMatrixTranslate(LocalTransform.matrix.getBatche(id), x, y);

    Shape.addComponent(world, id, ShapeKind.Triangle, point1[0], point1[1], point2[0], point2[1], point3[0], point3[1]);
    Color.addComponent(world, id, color[0], color[1], color[2], color[3]);
    Shadow.addComponent(world, id, shadow?.[0] ?? 0, shadow?.[1] ?? 0);
    Roundness.addComponent(world, id, roundness ?? 0);

    return id;
}