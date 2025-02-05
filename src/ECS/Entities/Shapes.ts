import { addComponent, addEntity, IWorld } from 'bitecs';
import { applyMatrixTranslate, resetMatrix, Transform } from '../Components/Transform.ts';
import { Color, Roundness, setColor } from '../Components/Common.ts';
import { setCircle, setRectangle, setTriangle, Shape } from '../Components/Shape.ts';

export function createCircle(world: IWorld, { x, y, radius, color }: {
    x: number;
    y: number;
    radius: number;
    color: [number, number, number, number];
}) {
    const id = addEntity(world);
    addComponent(world, Transform, id);
    addComponent(world, Roundness, id);
    addComponent(world, Color, id);
    addComponent(world, Shape, id);

    resetMatrix(id);
    applyMatrixTranslate(id, x, y);
    setCircle(id, radius);
    setColor(id, color[0], color[1], color[2], color[3]);

    return id;
}

export function createRectangle(world: IWorld, { x, y, width, height, color }: {
    x: number;
    y: number;
    width: number;
    height: number;
    color: [number, number, number, number];
}) {
    const id = addEntity(world);
    addComponent(world, Transform, id);
    addComponent(world, Roundness, id);
    addComponent(world, Color, id);
    addComponent(world, Shape, id);

    resetMatrix(id);
    applyMatrixTranslate(id, x, y);
    setRectangle(id, width, height);
    setColor(id, color[0], color[1], color[2], color[3]);

    return id;
}

export function createTriangle(world: IWorld, { x, y, color, point1, point2, point3 }: {
    x: number;
    y: number;
    point1: [number, number];
    point2: [number, number];
    point3: [number, number];
    color: [number, number, number, number];
}) {
    const id = addEntity(world);
    addComponent(world, Transform, id);
    addComponent(world, Roundness, id);
    addComponent(world, Color, id);
    addComponent(world, Shape, id);

    resetMatrix(id);
    applyMatrixTranslate(id, x, y);
    setTriangle(id, point1[0], point1[1], point2[0], point2[1], point3[0], point3[1]);
    setColor(id, color[0], color[1], color[2], color[3]);

    return id;
}