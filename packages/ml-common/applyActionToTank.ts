import { abs, sign } from '../../lib/math.ts';
import { ACTION_HEAD_DIMS } from '../ml/src/Models/Create.ts';
import { TankController } from '../tanks/src/Game/ECS/Components/TankController.ts';

export type Actions = Float32Array | [number, number, number, number];

export function applyActionToTank(
    tankEid: number,
    actions: Actions,
    isContinuous: boolean,
) {
    const shoot = isContinuous ? actions[0] : actions[0] - 0.5;
    const move = isContinuous ? actions[1] : toAction(actions, 1);
    const rotate = isContinuous ? actions[2] : toAction(actions, 2);
    const turretRot = isContinuous ? actions[3] : toAction(actions, 3);

    TankController.setShooting$(tankEid, shoot);
    TankController.setMove$(tankEid, move);
    TankController.setRotate$(tankEid, rotate);
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