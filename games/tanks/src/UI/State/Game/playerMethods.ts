import { addTank } from './engineMethods.ts';
import { toggleGame } from './GameState.ts';
import { destroyEngine } from './engine.ts';
import { activateBots, destroyBots, finalizeGameState, mapSlotToEid$ } from './gameMethods.ts';
import { TankType } from '../../../Game/ECS/Components/Tank.ts';
import { dedobs, DEDOBS_REMOVE_DELAY, DEDOBS_RESET_DELAY } from '../../../../../../lib/Rx/dedobs.ts';
import { map } from 'rxjs';

export const PLAYER_TEAM_ID = 0;

export const addTankToSlot = (tankType: TankType, slot: number) => {
    const eid = addTank(slot, PLAYER_TEAM_ID, tankType);
    mapSlotToEid$.next(mapSlotToEid$.value.set(slot, eid));
};

export const getTankEidBySlot = (slot: number) => {
    return mapSlotToEid$.value.get(slot);
};

export const getTankEidBySlot$ = dedobs(
    (slot: number) => {
        return mapSlotToEid$.pipe(
            map((map) => map.get(slot)),
        );
    },
    {
        removeDelay: DEDOBS_REMOVE_DELAY,
        resetDelay: DEDOBS_RESET_DELAY,
    },
);

export const mapAddTank = {
    [TankType.Light]: addTankToSlot.bind(null, TankType.Light),
    [TankType.Medium]: addTankToSlot.bind(null, TankType.Medium),
    [TankType.Heavy]: addTankToSlot.bind(null, TankType.Heavy),
};

export const startGame = async () => {
    await finalizeGameState();
    activateBots();
    toggleGame(true);
};

export const exitGame = () => {
    destroyBots();
    destroyEngine();
    toggleGame(false);
};
