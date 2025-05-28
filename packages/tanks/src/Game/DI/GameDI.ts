import { PhysicalWorld } from '../Physical/initPhysicalWorld.ts';
import { World } from '../../../../renderer/src/ECS/world.ts';
import { EntityId } from 'bitecs';

export const GameDI: {
    width: number;
    height: number;
    world: World;
    physicalWorld: PhysicalWorld;
    gameTick: (delta: number) => void;
    destroy: () => void;
    setRenderTarget: (canvas: undefined | null | HTMLCanvasElement) => void;
    enablePlayer: () => void
    setPlayerTank: (tankEid: null | EntityId) => void
} = {
    width: null as any,
    height: null as any,
    world: null as any,
    physicalWorld: null as any,
    gameTick: null as any,
    destroy: null as any,
    setRenderTarget: null as any,
    enablePlayer: null as any,
    setPlayerTank: null as any,
};
