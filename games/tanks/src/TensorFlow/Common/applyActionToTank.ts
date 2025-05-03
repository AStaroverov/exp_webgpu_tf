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
    const shoot = actions[0] > 0;
    const move = actions[1];
    const rotate = actions[2];
    const aimX = actions[3];
    const aimY = actions[4];

    TankController.setShooting$(tankEid, shoot);
    TankController.setMove$(tankEid, clamp(move * limitMove, -limitMove, limitMove));
    TankController.setRotate$(tankEid, clamp(rotate * limitRotation, -limitRotation, limitRotation));
    TankController.setTurretDir$(
        tankEid,
        clamp(aimX * limitAimDir, -limitAimDir, limitAimDir),
        clamp(aimY * limitAimDir, -limitAimDir, limitAimDir),
    );
}
