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
    TankController.setMove$(tankEid, clamp(move, -limitMove, limitMove));
    TankController.setRotate$(tankEid, clamp(rotate, -limitRotation, limitRotation));
    TankController.setTurretDir$(
        tankEid,
        clamp(aimX, -limitAimDir, limitAimDir),
        clamp(aimY, -limitAimDir, limitAimDir),
    );
}
