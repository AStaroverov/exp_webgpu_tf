import { EntityId } from 'bitecs';
import { PluginDI } from './PluginDI.ts';

export const GameDI: {
    width: number;
    height: number;
    gameTick: (delta: number) => void;
    destroy: () => void;
    enableSound: () => void;
    setRenderTarget: (canvas: undefined | null | HTMLCanvasElement) => void;
    setCameraTarget: (tankEid: null | EntityId) => void;

    plugins: typeof PluginDI
} = {
    width: null as any,
    height: null as any,
    gameTick: null as any,
    destroy: null as any,
    enableSound: null as any,
    setRenderTarget: null as any,
    setCameraTarget: null as any,

    plugins: PluginDI,
};
