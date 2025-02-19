import { addComponent, addEntity, World } from 'bitecs';
import { addTransformComponents, applyMatrixTranslate, LocalTransform } from '../Components/Transform.ts';
import { Color, Roundness, setColor, setShadow, Shadow, TColor, TShadow } from '../Components/Common.ts';
import { setCircle, setRectangle, setTriangle, Shape } from '../Components/Shape.ts';

export function createCircle(world: World, { x, y, radius, color, shadow }: {
    x: number;
    y: number;
    radius: number;
    color: TColor;
    shadow: TShadow;
}) {
    const id = addEntity(world);
    addComponent(world, id, Roundness);
    addComponent(world, id, Color);
    addComponent(world, id, Shape);
    addComponent(world, id, Shadow);

    addTransformComponents(world, id);
    applyMatrixTranslate(LocalTransform.matrix.getBatche(id), x, y);

    setCircle(id, radius);
    setColor(id, color[0], color[1], color[2], color[3]);
    setShadow(id, shadow[0], shadow[1]);

    return id;
}

export function createRectangle(world: World, { x, y, width, height, color, shadow }: {
    x: number;
    y: number;
    width: number;
    height: number;
    color: TColor;
    shadow: TShadow;
}) {
    const id = addEntity(world);
    addComponent(world, id, Roundness);
    addComponent(world, id, Color);
    addComponent(world, id, Shape);
    addComponent(world, id, Shadow);

    addTransformComponents(world, id);
    applyMatrixTranslate(LocalTransform.matrix.getBatche(id), x, y);

    setRectangle(id, width, height);
    setColor(id, color[0], color[1], color[2], color[3]);
    setShadow(id, shadow[0], shadow[1]);

    return id;
}

export function createTriangle(world: World, { x, y, color, point1, point2, point3, shadow }: {
    x: number;
    y: number;
    point1: [number, number];
    point2: [number, number];
    point3: [number, number];
    color: TColor;
    shadow: TShadow;
}) {
    const id = addEntity(world);
    addComponent(world, id, LocalTransform);
    addComponent(world, id, Roundness);
    addComponent(world, id, Color);
    addComponent(world, id, Shape);

    addTransformComponents(world, id);
    applyMatrixTranslate(LocalTransform.matrix.getBatche(id), x, y);

    setTriangle(id, point1[0], point1[1], point2[0], point2[1], point3[0], point3[1]);
    setColor(id, color[0], color[1], color[2], color[3]);
    setShadow(id, shadow[0], shadow[1]);

    return id;
}