import { EntityId } from 'bitecs';

export const PlayerEnvDI: {
    tankEid: null | EntityId,
    document: Document,
    window: Window,
    destroy: VoidFunction,
    inputFrame: VoidFunction,
} = {
    tankEid: null,
    document: null as any,
    window: null as any,
    destroy: null as any,
    inputFrame: null as any,
};
