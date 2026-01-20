import { EntityId, World } from 'bitecs';
import { PluginDI } from './PluginDI.js';

export const GameDI: {
    rows: number;
    cells: number;
    world: World;
    gameTick: (delta: number) => void;
    destroy: () => void;
    enableSound: () => void;
    setRenderTarget: (canvas: undefined | null | HTMLCanvasElement) => void;
    enablePlayer: () => void
    setPlayerId: (playerId: null | EntityId) => void

    plugins: typeof PluginDI
} = {
    cells: null as any,
    rows: null as any,
    world: null as any,
    gameTick: null as any,
    destroy: null as any,
    enableSound: null as any,
    setRenderTarget: null as any,
    enablePlayer: null as any,
    setPlayerId: null as any,

    plugins: PluginDI,
};
