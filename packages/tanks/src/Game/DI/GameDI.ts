import { PhysicalWorld } from '../Physical/initPhysicalWorld.ts';
import { World } from '../../../../renderer/src/ECS/world.ts';
import { EntityId } from 'bitecs';
import { PluginDI } from './PluginDI.ts';

export const GameDI: {
    width: number;
    height: number;
    world: World;
    physicalWorld: PhysicalWorld;
    gameTick: (delta: number) => void;
    destroy: () => void;
    enableSound: () => void;
    setRenderTarget: (canvas: undefined | null | HTMLCanvasElement) => void;
    enablePlayer: () => void
    setPlayerTank: (tankEid: null | EntityId) => void
    setCameraTarget: (tankEid: null | EntityId) => void
    setInfiniteMapMode: (enabled: boolean) => void

    plugins: typeof PluginDI
} = {
    width: null as any,
    height: null as any,
    world: null as any,
    physicalWorld: null as any,
    gameTick: null as any,
    destroy: null as any,
    enableSound: null as any,
    setRenderTarget: null as any,
    enablePlayer: null as any,
    setPlayerTank: null as any,
    setCameraTarget: null as any,
    setInfiniteMapMode: null as any,

    plugins: PluginDI,
};
