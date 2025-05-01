import { TankController } from '../../ECS/Components/TankController.ts';
import { Actions, readActions } from './actions.ts';
import { clamp } from 'lodash-es';

export function applyActionToTank(
    tankEid: number,
    action: Actions,
    limitMove = 1,
    limitRotation = 1,
    limitAimDir = 2,
) {
    const { shoot, move, rotate, aimX, aimY } = readActions(action);

    TankController.setShooting$(tankEid, shoot);
    TankController.setMove$(tankEid, clamp(TankController.move[tankEid] + move * 0.5, -limitMove, limitMove));
    TankController.setRotate$(tankEid, clamp(TankController.rotation[tankEid] + rotate * 0.5, -limitRotation, limitRotation));
    TankController.setTurretDir$(
        tankEid,
        clamp(TankController.turretDir.get(tankEid, 0) + aimX * 0.5, -limitAimDir, limitAimDir),
        clamp(TankController.turretDir.get(tankEid, 1) + aimY * 0.5, -limitAimDir, limitAimDir),
    );
}
