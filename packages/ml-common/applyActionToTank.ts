import { abs, sign } from '../../lib/math.ts';
import { ACTION_HEAD_DIMS } from '../ml/src/Models/Create.ts';
import { TankController } from '../tanks/src/Game/ECS/Components/TankController.ts';

export type Actions = Float32Array | [number, number, number, number];

export function applyActionToTank(
    tankEid: number,
    actions: Actions,
    isContinuous: boolean,
) {
    const move = isContinuous ? actions[0] : toAction(actions, 0);
    const rotate = isContinuous ? actions[1] : toAction(actions, 1);
    
    const turretShoot = isContinuous ? actions[2] : actions[2] - 0.5;
    const turretRot = isContinuous ? actions[3] : toAction(actions, 3);

    TankController.setMove$(tankEid, move);
    TankController.setRotate$(tankEid, rotate);
    
    TankController.setShooting$(tankEid, turretShoot);
    TankController.setTurretRotation$(tankEid, turretRot);
}

function toAction(actions: Actions, index: number): number {
    const dims = (ACTION_HEAD_DIMS[index]-1) / 2;
    const action = actions[index];
    const value  = sign(action - dims) * (abs(action - dims) / dims) ** 2;
    return value;
}

// console.log('>>>>>>>>>')
// new Array(31).fill(0).forEach((_,i) => {
//     console.log(toAction([0,0,0,i], 3))
// });
// debugger;