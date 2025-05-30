import { GameDI } from '../../Game/DI/GameDI.ts';
import { getAlivePilots } from '../Components/Pilot.ts';
import { SNAPSHOT_EVERY } from '../../TensorFlow/Common/consts.ts';
import { snapshotTankInputTensor } from '../Utils/snapshotTankInputTensor.ts';
import { PilotsState } from '../Singelton/PilotsState.ts';
import { max } from '../../../../../lib/math.ts';

export function createPilotSystem(game = GameDI) {
    let frame = 0;

    return () => {
        if (!PilotsState.enabled) return;

        frame++;

        const alivePilots = getAlivePilots();

        if (!alivePilots.every(a => a.isReady())) return;

        const snapshotEvery = max(SNAPSHOT_EVERY, alivePilots.length);

        for (let i = 0; i < alivePilots.length; i++) {
            const pilot = alivePilots[i];
            const shouldAction = (i + frame) % snapshotEvery === 0;

            if (!shouldAction || !pilot.isReady()) continue;

            pilot.evaluateTankBehaviour?.(game.width, game.height);
            snapshotTankInputTensor(pilot.tankEid);
            pilot.updateTankBehaviour(game.width, game.height);
        }
    };
}
