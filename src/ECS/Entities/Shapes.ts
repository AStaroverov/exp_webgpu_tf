import { addEntity, World } from 'bitecs';
import { addTransformComponents, applyMatrixTranslate, LocalTransform } from '../Components/Transform.ts';
import { ColorMethods, RoundnessMethods, ShadowMethods, TColor, TShadow } from '../Components/Common.ts';
import { ShapeKind, ShapeMethods } from '../Components/Shape.ts';

export function createCircle(world: World, { x, y, radius, color, shadow }: {
    x: number;
    y: number;
    radius: number;
    color: TColor;
    shadow: TShadow;
}) {
    const id = addEntity(world);

    addTransformComponents(world, id);
    applyMatrixTranslate(LocalTransform.matrix.getBatche(id), x, y);

    ShapeMethods.addComponent(id, ShapeKind.Circle, radius);
    ColorMethods.addComponent(id, color[0], color[1], color[2], color[3]);
    ShadowMethods.addComponent(id, shadow[0], shadow[1]);

    return id;
}

export function createRectangle(world: World, { x, y, width, height, color, roundness, shadow }: {
    x: number;
    y: number;
    width: number;
    height: number;
    roundness: number;
    color: TColor;
    shadow: TShadow;
}) {
    const id = addEntity(world);

    addTransformComponents(world, id);
    applyMatrixTranslate(LocalTransform.matrix.getBatche(id), x, y);

    ShapeMethods.addComponent(id, ShapeKind.Rectangle, width, height);
    ColorMethods.addComponent(id, color[0], color[1], color[2], color[3]);
    ShadowMethods.addComponent(id, shadow[0], shadow[1]);
    RoundnessMethods.addComponent(id, roundness);

    return id;
}

export function createTriangle(world: World, { x, y, color, roundness, point1, point2, point3, shadow }: {
    x: number;
    y: number;
    roundness: number;
    point1: [number, number];
    point2: [number, number];
    point3: [number, number];
    color: TColor;
    shadow: TShadow;
}) {
    const id = addEntity(world);

    addTransformComponents(world, id);
    applyMatrixTranslate(LocalTransform.matrix.getBatche(id), x, y);

    ShapeMethods.addComponent(id, ShapeKind.Triangle, point1[0], point1[1], point2[0], point2[1], point3[0], point3[1]);
    ColorMethods.addComponent(id, color[0], color[1], color[2], color[3]);
    ShadowMethods.addComponent(id, shadow[0], shadow[1]);
    RoundnessMethods.addComponent(id, roundness);

    return id;
}