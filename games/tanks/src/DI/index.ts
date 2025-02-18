import { PhysicalWorld } from '../index.ts';
import { World } from '../../../../src/ECS/world.ts';

export const DI: {
    document: Document;
    canvas: HTMLCanvasElement;
    world: World;
    physicalWorld: PhysicalWorld;
    gameTick: (delta: number) => void;
} = {
    document: window.document,
    canvas: null as any,
    world: null as any,
    physicalWorld: null as any,
    gameTick: null as any,
};