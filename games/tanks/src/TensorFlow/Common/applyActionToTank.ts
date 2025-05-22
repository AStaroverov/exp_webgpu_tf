import { TankController } from '../../Game/ECS/Components/TankController.ts';
import { clamp } from 'lodash-es';

export type Actions = Float32Array | [number, number, number, number, number];


export function applyActionToTank(
    tankEid: number,
    actions: Actions,
    limitMove = 1,
    limitRotation = 1,
    limitAimDir = 2,
) {
    TankController.setShooting$(tankEid, clamp(actions[0], -1, 1));
    TankController.setMove$(tankEid, clamp(actions[1], -limitMove, limitMove));
    TankController.setRotate$(tankEid, clamp(actions[2], -limitRotation, limitRotation));
    TankController.setTurretDir$(tankEid,
        clamp(actions[3], -limitAimDir, limitAimDir),
        clamp(actions[4], -limitAimDir, limitAimDir),
    );
}
