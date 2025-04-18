import { TankController } from '../../ECS/Components/TankController.ts';
import { Actions, readActions } from './actions.ts';
import { clamp } from 'lodash-es';

export function applyActionToTank(tankEid: number, action: Actions) {
    const { shoot, move, rotate, aimX, aimY } = readActions(action);

    TankController.setShooting$(tankEid, shoot);
    TankController.setMove$(tankEid, clamp(TankController.move[tankEid] + move * 0.5, -0.8, 0.8));
    TankController.setRotate$(tankEid, clamp(TankController.rotation[tankEid] + rotate * 0.5, -0.8, 0.8));
    TankController.setTurretDir$(
        tankEid,
        clamp(TankController.turretDir.get(tankEid, 0) + aimX * 0.5, -1.2, 1.2),
        clamp(TankController.turretDir.get(tankEid, 1) + aimY * 0.5, -1.2, 1.2),
    );
}
