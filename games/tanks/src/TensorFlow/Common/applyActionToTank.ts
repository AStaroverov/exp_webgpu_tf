import { TankController } from '../../ECS/Components/TankController.ts';
import { Actions, readAction } from './readAction.ts';

export function applyActionToTank(tankEid: number, action: Actions) {
    const { shoot, move, rotate, aimX, aimY } = readAction(action);

    TankController.setShooting$(tankEid, shoot);
    TankController.setMove$(tankEid, move);
    TankController.setRotate$(tankEid, rotate);
    TankController.setTurretDir$(tankEid, aimX, aimY);
}
