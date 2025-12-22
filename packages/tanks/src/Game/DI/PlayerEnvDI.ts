import { EntityId } from 'bitecs';

// Real pearson player info
export const PlayerEnvDI: {
    playerId: null | EntityId,
    tankEid: null | EntityId,
    eventTarget: HTMLElement,
    destroy: VoidFunction,
    inputFrame: VoidFunction,
} = {
    playerId: null,
    tankEid: null,
    eventTarget: null as any,
    destroy: null as any,
    inputFrame: null as any,
};
