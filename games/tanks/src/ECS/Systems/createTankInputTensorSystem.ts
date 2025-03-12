import { DI } from '../../DI';
import { getTankHealth, Tank } from '../Components/Tank.ts';
import {
    TANK_INPUT_TENSOR_MAX_BULLETS,
    TANK_INPUT_TENSOR_MAX_ENEMIES,
    TankInputTensor,
} from '../Components/TankState.ts';
import { getEntityIdByPhysicalId, RigidBodyState } from '../Components/Physical.ts';
import { hypot, max } from '../../../../../lib/math.ts';
import { Ball, Collider } from '@dimforge/rapier2d';
import { CollisionGroup, createCollisionGroups } from '../../Physical/createRigid.ts';
import { query } from 'bitecs';
import { Player } from '../Components/Player.ts';
import { getMatrixTranslation, LocalTransform } from '../../../../../src/ECS/Components/Transform.ts';

export function createTankInputTensorSystem(options = DI) {
    const { world } = options;
    const colliderIds = new Float64Array(max(TANK_INPUT_TENSOR_MAX_ENEMIES, TANK_INPUT_TENSOR_MAX_BULLETS));

    return () => {
        if (!options.shouldCollectTensor) return;

        const tankEids = query(world, [Tank, TankInputTensor, RigidBodyState]);

        TankInputTensor.resetEnemiesCoords();
        TankInputTensor.resetBulletsCoords();

        for (let i = 0; i < tankEids.length; i++) {
            const tankEid = tankEids[i];
            const health = getTankHealth(tankEid);
            const linvel = RigidBodyState.linvel.getBatche(tankEid);
            const position = RigidBodyState.position.getBatche(tankEid);
            const aimLocal = LocalTransform.matrix.getBatche(Tank.aimEid[tankEid]);

            TankInputTensor.setTankData(
                tankEid,
                health,
                linvel,
                position,
                getMatrixTranslation(aimLocal),
            );

            // Find enemies
            const [enemyCount] = findTankEnemies(tankEid, colliderIds);

            for (let j = 0; j < enemyCount; j++) {
                const enemyEid = colliderIds[j];

                TankInputTensor.setEnemiesData(
                    tankEid,
                    j,
                    enemyEid,
                    getTankHealth(enemyEid),
                    RigidBodyState.position.getBatche(enemyEid),
                    RigidBodyState.linvel.getBatche(enemyEid),
                );
            }

            const [bulletCount] = findTankDangerBullets(tankEid, colliderIds);

            for (let j = 0; j < bulletCount; j++) {
                const bulletEid = colliderIds[j];

                TankInputTensor.setBulletsData(
                    tankEid,
                    j,
                    bulletEid,
                    RigidBodyState.position.getBatche(bulletEid),
                    RigidBodyState.linvel.getBatche(bulletEid),
                );
            }
        }
    };
}

export function findTankEnemies(tankEid: number, out = new Float64Array(TANK_INPUT_TENSOR_MAX_ENEMIES), { physicalWorld } = DI) {
    // Find enemies
    const position = {
        x: RigidBodyState.position.get(tankEid, 0),
        y: RigidBodyState.position.get(tankEid, 1),
    };
    const rotation = RigidBodyState.rotation[tankEid];

    let enemyCount = 0;

    for (let j = 1; j < 5; j++) {
        const radius = 10 ** j;
        physicalWorld.intersectionsWithShape(
            position,
            rotation,
            new Ball(radius),
            (collider: Collider) => {
                const eid = getEntityIdByPhysicalId(collider.handle);
                if (eid !== 0 && tankEid !== eid) {
                    out[enemyCount++] = eid;
                }

                return enemyCount < TANK_INPUT_TENSOR_MAX_ENEMIES;
            },
            undefined,
            createCollisionGroups(CollisionGroup.TANK_BASE, CollisionGroup.TANK_BASE),
        );
        if (enemyCount >= TANK_INPUT_TENSOR_MAX_ENEMIES) {
            break;
        }
    }

    return [enemyCount, out] as const;
}

export function findTankDangerBullets(tankEid: number, out = new Float64Array(TANK_INPUT_TENSOR_MAX_BULLETS), { physicalWorld } = DI) {
    let bulletCount = 0;
    const playerId = Player.id[tankEid];
    const position = {
        x: RigidBodyState.position.get(tankEid, 0),
        y: RigidBodyState.position.get(tankEid, 1),
    };
    const rotation = RigidBodyState.rotation[tankEid];

    for (let j = 1; j < 5; j++) {
        const radius = 10 ** j;
        physicalWorld.intersectionsWithShape(
            position,
            rotation,
            new Ball(radius),
            (collider: Collider) => {
                const eid = getEntityIdByPhysicalId(collider.handle);
                const vel = collider.parent()?.linvel();

                if (eid !== 0 && Player.id[eid] !== playerId && vel != null && hypot(vel.x, vel.y) > 100) {
                    out[bulletCount++] = eid;
                }

                return bulletCount < TANK_INPUT_TENSOR_MAX_BULLETS;
            },
            undefined,
            createCollisionGroups(CollisionGroup.BULLET, CollisionGroup.BULLET),
        );
        if (bulletCount >= TANK_INPUT_TENSOR_MAX_BULLETS) {
            break;
        }
    }

    return [bulletCount, out] as const;
}
