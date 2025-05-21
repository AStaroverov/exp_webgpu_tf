import { GameDI } from '../../../DI/GameDI.ts';
import { query } from 'bitecs';
import { Tank } from '../../Components/Tank.ts';
import { TankController } from '../../Components/TankController.ts';
import {
    getMatrixTranslationX,
    getMatrixTranslationY,
    LocalTransform,
    setMatrixTranslate,
} from '../../../../../../../src/ECS/Components/Transform.ts';
import { ZIndex } from '../../../consts.ts';
import { dist2, hypot } from '../../../../../../../lib/math.ts';
import { RigidBodyState } from '../../Components/Physical.ts';

const MAX_DIST = 800;

export function createTankAimSystem({ world } = GameDI) {
    return ((delta: number) => {
        const tankEids = query(world, [Tank, TankController]);

        for (let i = 0; i < tankEids.length; i++) {
            const tankEid = tankEids[i];
            const aimEid = Tank.aimEid[tankEid];
            const tankPos = RigidBodyState.position.getBatch(tankEid);
            const aimLocal = LocalTransform.matrix.getBatch(aimEid);
            const aimDir = TankController.turretDir.getBatch(tankEid);
            let aimX = getMatrixTranslationX(aimLocal) + aimDir[0] * delta * 0.3;
            let aimY = getMatrixTranslationY(aimLocal) + aimDir[1] * delta * 0.3;

            if (dist2(tankPos[0], tankPos[1], aimX, aimY) > MAX_DIST) {
                const dX = aimX - tankPos[0];
                const dY = aimY - tankPos[1];
                const dist = hypot(dX, dY);
                const nX = dX / dist;
                const nY = dY / dist;
                aimX = tankPos[0] + nX * MAX_DIST;
                aimY = tankPos[1] + nY * MAX_DIST;
            }

            setMatrixTranslate(aimLocal, aimX, aimY, ZIndex.Bullet);
        }
    });
}