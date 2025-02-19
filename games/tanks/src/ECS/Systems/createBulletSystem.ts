import { DI } from '../../DI';
import { spawnBullet } from '../Components/Bullet.ts';
import { Tank } from '../Components/Tank.ts';
import { TankController, TankControllerMethods } from '../Components/TankController.ts';
import { onSet, query } from 'bitecs';
import { Player } from '../Components/Player.ts';
import { GlobalTransform } from '../../../../../src/ECS/Components/Transform.ts';
import { createChangedDetector } from '../../../../../src/ECS/Systems/ChangedDetectorSystem.ts';

export function createSpawnerBulletsSystem({ world } = DI) {
    const tankControllerChanges = createChangedDetector(world, [onSet(TankController)]);

    return (() => {
        const tankEids = query(world, [Player, GlobalTransform, Tank, TankController]);

        for (let i = 0; i < tankEids.length; i++) {
            const tankEid = tankEids[i];

            if (tankControllerChanges.has(tankEid) && TankControllerMethods.shouldShot(tankEid)) {
                spawnBullet(tankEid);
                TankControllerMethods.resetShot$(tankEid);
            }
        }
    });
}
