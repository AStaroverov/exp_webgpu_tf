import { DI } from '../../DI';
import { query } from 'bitecs';
import { Tank } from '../Components/Tank.ts';
import { TankController } from '../Components/TankController.ts';
import { GlobalTransform, setMatrixTranslate } from '../../../../../src/ECS/Components/Transform.ts';
import { ZIndex } from '../../consts.ts';

export function createAimSystem({ world } = DI) {
    return (() => {
        const tankEids = query(world, [Tank, TankController]);

        for (let i = 0; i < tankEids.length; i++) {
            const tankEid = tankEids[i];
            const eimEid = Tank.aimEid[tankEid];
            const turretTarget = TankController.turretTarget.getBatche(tankEid);

            setMatrixTranslate(GlobalTransform.matrix.getBatche(eimEid), turretTarget[0], turretTarget[1], ZIndex.Bullet);
        }
    });
}