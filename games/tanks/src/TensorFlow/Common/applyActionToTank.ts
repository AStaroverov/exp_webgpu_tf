import { TankController } from '../../ECS/Components/TankController.ts';
import { clamp } from 'lodash-es';
import { PI } from '../../../../../lib/math.ts';

export type Actions = Float32Array | [number, number, number, number, number];

export function applyActionToTank(
    tankEid: number,
    actions: Actions,
    limitMove = 1,
    limitRotation = PI,
    limitAimDir = 2,
) {
    const shoot = actions[0];
    const move = actions[1];
    const rotate = actions[2];
    const aimX = actions[3];
    const aimY = actions[4];

    TankController.setShooting$(tankEid, shoot > 0);
    TankController.setMove$(tankEid, clamp(TankController.move[tankEid] + move, -limitMove, limitMove));
    TankController.setRotate$(tankEid, clamp(TankController.rotation[tankEid] + rotate, -limitRotation, limitRotation));
    TankController.setTurretDir$(
        tankEid,
        clamp(TankController.turretDir.get(tankEid, 0) + aimX, -limitAimDir, limitAimDir),
        clamp(TankController.turretDir.get(tankEid, 1) + aimY, -limitAimDir, limitAimDir),
    );
}
