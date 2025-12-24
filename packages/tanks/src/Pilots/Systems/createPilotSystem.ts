import { isFunction } from 'lodash';
import { SNAPSHOT_EVERY } from '../../../../ml-common/consts.ts';
import { GameDI } from '../../Game/DI/GameDI.ts';
import { TankAgent } from '../Agents/CurrentActorAgent.ts';
import { getAlivePilotAgents, getPilotAgents } from '../Components/Pilot.ts';
import { PilotsState } from '../Singelton/PilotsState.ts';
import { snapshotTankInputTensor } from '../Utils/snapshotTankInputTensor.ts';

export function createPilotSystem() {
    let frame = 0;
    let currentPilots = [] as TankAgent[];

    return () => {
        if (!PilotsState.enabled) return;
        if (!getPilotAgents().every(isSynced)) return;

        const shouldAction = frame++ % SNAPSHOT_EVERY === 0;

        if (shouldAction) {
            for (const agent of currentPilots) {
                agent.evaluateTankBehaviour?.(GameDI.width, GameDI.height, frame, 0);
            }
        }

        if (shouldAction) {
            snapshotTankInputTensor();

            currentPilots = getAlivePilotAgents();

            for (const agent of currentPilots) {
                agent.scheduleUpdateTankBehaviour(GameDI.width, GameDI.height, frame);
            }

            for (const agent of currentPilots) {
                agent.applyUpdateTankBehaviour();
            }
        }
    };
}

function isSynced(agent: TankAgent): boolean {
    return isFunction(agent.isSynced) ? agent.isSynced() : true;
}