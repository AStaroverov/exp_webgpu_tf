import { DI } from '../../DI';
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

export function createAimSystem({ world } = DI) {
    return ((delta: number) => {
        const tankEids = query(world, [Tank, TankController]);

        for (let i = 0; i < tankEids.length; i++) {
            const tankEid = tankEids[i];
            const aimEid = Tank.aimEid[tankEid];
            const local = LocalTransform.matrix.getBatche(aimEid);
            const turretDir = TankController.turretDir.getBatche(tankEid);
            const turretTargetX = getMatrixTranslationX(local) + turretDir[0] * delta * 0.1;
            const turretTargetY = getMatrixTranslationY(local) + turretDir[1] * delta * 0.1;

            setMatrixTranslate(LocalTransform.matrix.getBatche(aimEid), turretTargetX, turretTargetY, ZIndex.Bullet);
        }
    });
}