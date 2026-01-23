import { isFunction } from 'lodash';
import { SNAPSHOT_EVERY } from '../../../../../../ml-common/consts';
import { GameDI } from '../../../GameEngine/DI/GameDI.js';
import { TankAgent } from '../Agents/CurrentActorAgent.js';
import { getAlivePilotAgents, getPilotAgents } from '../Components/Pilot.js';
import { PilotsState } from '../Singelton/PilotsState.js';
import { snapshotTankInputTensor } from '../Utils/snapshotTankInputTensor.js';

export function createPilotSystem() {
    let frame = 0;
    let currentPilots = [] as TankAgent[];

    return () => {
        if (!PilotsState.enabled) return;
        if (!getPilotAgents().every(isSynced)) return;
        if (frame++ % SNAPSHOT_EVERY !== 0) return;
        
        for (const agent of currentPilots) {
            agent.evaluateTankBehaviour?.(GameDI.cells, GameDI.rows, frame);
        }

        snapshotTankInputTensor();

        currentPilots = getAlivePilotAgents();

        for (const agent of currentPilots) {
            agent.scheduleUpdateTankBehaviour(GameDI.cells, GameDI.rows, frame);
        }

        for (const agent of currentPilots) {
            agent.applyUpdateTankBehaviour(GameDI.cells, GameDI.rows, frame);
        }
    };
}

function isSynced(agent: TankAgent): boolean {
    return isFunction(agent.isSynced) ? agent.isSynced() : true;
}