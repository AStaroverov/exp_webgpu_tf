import { TankController } from '../../ECS/Components/TankController.ts';
import { Actions, readActions } from './actions.ts';
import { clamp } from 'lodash-es';

export function applyActionToTank(tankEid: number, action: Actions) {
    const { shoot, move, rotate, aimX, aimY } = readActions(action);

    TankController.setShooting$(tankEid, shoot);
    TankController.setMove$(tankEid, clamp(TankController.move[tankEid] + move * 0.3, -0.3, 0.3));
    TankController.setRotate$(tankEid, clamp(TankController.rotation[tankEid] + rotate * 0.3, -0.3, 0.3));
    TankController.setTurretDir$(
        tankEid,
        clamp(TankController.turretDir.get(tankEid, 0) + aimX * 0.3, -0.4, 0.4),
        clamp(TankController.turretDir.get(tankEid, 1) + aimY * 0.3, -0.4, 0.4),
    );
}
