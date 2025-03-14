import { PhysicalWorld } from '../index.ts';
import { World } from '../../../../src/ECS/world.ts';

export const GameDI: {
    width: number;
    height: number;
    world: World;
    physicalWorld: PhysicalWorld;
    shouldCollectTensor: boolean
    gameTick: (delta: number, withDraw?: boolean) => void;
    destroy: () => void;
} = {
    width: null as any,
    height: null as any,
    world: null as any,
    physicalWorld: null as any,
    shouldCollectTensor: false,
    gameTick: null as any,
    destroy: null as any,
};
