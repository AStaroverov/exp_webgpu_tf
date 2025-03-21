import { TankController } from '../../ECS/Components/TankController.ts';
import { Actions, readActions } from './actions.ts';

export function applyActionToTank(tankEid: number, action: Actions) {
    const { shoot, move, rotate, aimX, aimY } = readActions(action);

    TankController.setShooting$(tankEid, shoot);
    TankController.setMove$(tankEid, move);
    TankController.setRotate$(tankEid, rotate);
    TankController.setTurretDir$(tankEid, aimX, aimY);
}
