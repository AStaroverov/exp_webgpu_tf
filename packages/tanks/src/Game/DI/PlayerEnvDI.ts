import { EntityId } from 'bitecs';

// Real pearson player info
export const PlayerEnvDI: {
    playerId: null | EntityId,
    tankEid: null | EntityId,
    document: Document,
    window: Window,
    destroy: VoidFunction,
    inputFrame: VoidFunction,
} = {
    playerId: null,
    tankEid: null,
    document: null as any,
    window: null as any,
    destroy: null as any,
    inputFrame: null as any,
};
