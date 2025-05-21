import { TankController } from '../../Game/ECS/Components/TankController.ts';
import { clamp } from 'lodash-es';

export type Actions = Float32Array | [number, number, number, number, number];

const defaultProbability = [1, 1, 1, 1, 1] as Actions;

export function applyActionToTank(
    tankEid: number,
    actions: Actions,
    probability = defaultProbability, // 0..1, 0 - not sure, 1 - sure
    limitMove = 1,
    limitRotation = 1,
    limitAimDir = 2,
) {
    const shoot = blendLinear(
        TankController.shoot[tankEid],
        actions[0],
        probability[0],
        1,
    );
    const move = blendLinear(
        TankController.move[tankEid],
        actions[1],
        probability[1],
        limitMove,
    );
    const rotate = blendLinear(
        TankController.rotation[tankEid],
        actions[2],
        probability[2],
        limitRotation,
    );
    const aimX = blendLinear(
        TankController.turretDir.get(tankEid, 0),
        actions[3],
        probability[3],
        limitAimDir,
    );
    const aimY = blendLinear(
        TankController.turretDir.get(tankEid, 1),
        actions[4],
        probability[4],
        limitAimDir,
    );
    TankController.setShooting$(tankEid, shoot);
    TankController.setMove$(tankEid, move);
    TankController.setRotate$(tankEid, rotate);
    TankController.setTurretDir$(tankEid, aimX, aimY);
}

function blendLinear(prev: number, raw: number, p: number, limit: number) {
    const next = prev * (1 - p) + raw * limit * p;
    return clamp(next, -limit, limit);
}