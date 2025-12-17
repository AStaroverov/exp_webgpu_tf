import { map } from 'rxjs';
import { dedobs, DEDOBS_REMOVE_DELAY, DEDOBS_RESET_DELAY } from '../../../../../../lib/Rx/dedobs.ts';
import { initTensorFlow } from '../../../../../ml-common/initTensorFlow.ts';
import { TankVehicleType } from '../../../Game/ECS/Entities/Tank/createTank.ts';
import { destroyEngine } from './engine.ts';
import { addTank } from './engineMethods.ts';
import { activateBots, deactivateBots, finalizeGameState, mapSlotToEid$ } from './gameMethods.ts';
import { toggleGame } from './GameState.ts';

export const PLAYER_TEAM_ID = 0;

export const addTankToSlot = (vehicleType: TankVehicleType, slot: number) => {
    const eid = addTank(slot, PLAYER_TEAM_ID, vehicleType);
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

export const startGame = async () => {
    await initTensorFlow('wasm');
    await finalizeGameState();
    activateBots();
    toggleGame(true);
};

export const exitGame = () => {
    deactivateBots();
    destroyEngine();
    toggleGame(false);
};
