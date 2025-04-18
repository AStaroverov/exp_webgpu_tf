import { GameDI } from '../../DI/GameDI.ts';
import { query } from 'bitecs';
import { Tank } from '../Components/Tank.ts';
import { TankController } from '../Components/TankController.ts';
import {
    getMatrixTranslationX,
    getMatrixTranslationY,
    LocalTransform,
    setMatrixTranslate,
} from '../../../../../src/ECS/Components/Transform.ts';
import { ZIndex } from '../../consts.ts';
import { dist2, hypot } from '../../../../../lib/math.ts';
import { RigidBodyState } from '../Components/Physical.ts';

const MAX_DIST = 1200;

export function createAimSystem({ world } = GameDI) {
    return ((delta: number) => {
        const tankEids = query(world, [Tank, TankController]);

        for (let i = 0; i < tankEids.length; i++) {
            const tankEid = tankEids[i];
            const aimEid = Tank.aimEid[tankEid];
            const tankPos = RigidBodyState.position.getBatch(tankEid);
            const turretLocal = LocalTransform.matrix.getBatch(aimEid);
            const turretDir = TankController.turretDir.getBatch(tankEid);
            let turretTargetX = getMatrixTranslationX(turretLocal) + turretDir[0] * delta * 0.1;
            let turretTargetY = getMatrixTranslationY(turretLocal) + turretDir[1] * delta * 0.1;

            if (dist2(tankPos[0], tankPos[1], turretTargetX, turretTargetY) > MAX_DIST) {
                const dX = turretTargetX - tankPos[0];
                const dY = turretTargetY - tankPos[1];
                const dist = hypot(dX, dY);
                const nX = dX / dist;
                const nY = dY / dist;
                turretTargetX = tankPos[0] + nX * MAX_DIST;
                turretTargetY = tankPos[1] + nY * MAX_DIST;
            }

            setMatrixTranslate(LocalTransform.matrix.getBatch(aimEid), turretTargetX, turretTargetY, ZIndex.Bullet);
        }
    });
}