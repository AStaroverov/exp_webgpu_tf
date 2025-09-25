import { GameDI } from '../../Game/DI/GameDI.ts';
import { SNAPSHOT_EVERY } from '../../TensorFlow/Common/consts.ts';
import { TankAgent } from '../Agents/CurrentActorAgent.ts';
import { getAlivePilots, Pilot } from '../Components/Pilot.ts';
import { PilotsState } from '../Singelton/PilotsState.ts';
import { snapshotTankInputTensor } from '../Utils/snapshotTankInputTensor.ts';

export function createPilotSystem() {
    let frame = 0;
    let currentPilots = [] as TankAgent[];

    return () => {
        if (!PilotsState.enabled || !Pilot.isSynced()) return;

        const shouldAction = frame++ % SNAPSHOT_EVERY === 0;

        if (shouldAction) {
            for (const agent of currentPilots) {
                agent.evaluateTankBehaviour?.(GameDI.width, GameDI.height, frame);
            }
        }

        if (shouldAction) {
            snapshotTankInputTensor();

            currentPilots = getAlivePilots();

            for (const agent of currentPilots) {
                agent.updateTankBehaviour(GameDI.width, GameDI.height, frame);
            }
        }
    };
}
