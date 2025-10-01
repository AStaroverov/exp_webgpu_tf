import { clamp } from 'lodash-es';
import { TankController } from '../../../tanks/src/Game/ECS/Components/TankController.ts';

export type Actions = Float32Array | [number, number, number, number];

const defaultProbability = [1, 1, 1, 1] as Actions;

export function applyActionToTank(
    tankEid: number,
    actions: Actions,
    probability = defaultProbability, // 0..1, 0 - not sure, 1 - sure
) {
    const shoot = blendLinear(
        TankController.shoot[tankEid],
        actions[0],
        probability[0],
    );
    const move = blendLinear(
        TankController.move[tankEid],
        actions[1],
        probability[1],
    );
    const rotate = blendLinear(
        TankController.rotation[tankEid],
        actions[2],
        probability[2],
    );

    const turretRot = blendLinear(
        TankController.turretRotation[tankEid],
        actions[3],
        probability[3],
    );

    TankController.setShooting$(tankEid, shoot);
    TankController.setMove$(tankEid, move);
    TankController.setRotate$(tankEid, rotate);
    TankController.setTurretRotation$(tankEid, turretRot);
}

function blendLinear(prev: number, raw: number, p: number): number {
    const next = prev * (1 - p) + raw * p;
    return clamp(next, -1, 1);
}