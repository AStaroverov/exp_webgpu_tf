import { TankController } from '../../ECS/Components/TankController.ts';
import { clamp } from 'lodash-es';

export type Actions = Float32Array | [number, number, number, number, number];

const defaultProbability = [1, 1, 1, 1, 1] as Actions;

export function applyActionToTank(
    tankEid: number,
    actions: Actions,
    probability = defaultProbability, // 0..1, 0 - not sure, 1 - sure
    limitMove = 1,
    limitRotation = 1,
    limitAimDir = 2,
) {
    const shoot = (1 - probability[0]) * TankController.shoot[tankEid] + (probability[0] * actions[0]);
    const move = (1 - probability[1]) * TankController.move[tankEid] + (limitMove * probability[1] * actions[1]);
    const rotate = (1 - probability[2]) * TankController.rotation[tankEid] + (limitRotation * probability[2] * actions[2]);
    const aimX = (1 - probability[3]) * TankController.turretDir.get(tankEid, 0) + (limitAimDir * probability[3] * actions[3]);
    const aimY = (1 - probability[4]) * TankController.turretDir.get(tankEid, 1) + (limitAimDir * probability[4] * actions[4]);

    TankController.setShooting$(tankEid, clamp(shoot, -1, 1));
    TankController.setMove$(tankEid, clamp(move, -limitMove, limitMove));
    TankController.setRotate$(tankEid, clamp(rotate, -limitRotation, limitRotation));
    TankController.setTurretDir$(
        tankEid,
        clamp(aimX, -limitAimDir, limitAimDir),
        clamp(aimY, -limitAimDir, limitAimDir),
    );
}
