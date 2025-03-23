import { GameDI } from '../../../DI/GameDI.ts';
import { getTankHealth, Tank, TANK_APPROXIMATE_COLLISION_RADIUS } from '../../Components/Tank.ts';
import {
    TANK_INPUT_TENSOR_MAX_BULLETS,
    TANK_INPUT_TENSOR_MAX_ENEMIES,
    TankInputTensor,
} from '../../Components/TankState.ts';
import { getEntityIdByPhysicalId, RigidBodyState } from '../../Components/Physical.ts';
import { hypot } from '../../../../../../lib/math.ts';
import { Ball, Collider } from '@dimforge/rapier2d';
import { CollisionGroup, createCollisionGroups } from '../../../Physical/createRigid.ts';
import { EntityId, query } from 'bitecs';
import { Player } from '../../Components/Player.ts';
import { getMatrixTranslation, LocalTransform } from '../../../../../../src/ECS/Components/Transform.ts';
import { hasIntersectionVectorAndCircle } from '../../../Utils/intersections.ts';

export function createTankInputTensorSystem(options = GameDI) {
    const { world } = options;

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
            const enemiesEids = Array.from(findTankEnemies(tankEid));

            for (let j = 0; j < enemiesEids.length; j++) {
                const enemyEid = enemiesEids[j];

                TankInputTensor.setEnemiesData(
                    tankEid,
                    j,
                    enemyEid,
                    RigidBodyState.position.getBatche(enemyEid),
                    RigidBodyState.linvel.getBatche(enemyEid),
                    getMatrixTranslation(LocalTransform.matrix.getBatche(Tank.aimEid[enemyEid])),
                );
            }

            const bulletsEids = Array.from(findTankDangerBullets(tankEid));

            for (let j = 0; j < bulletsEids.length; j++) {
                const bulletEid = bulletsEids[j];

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

export function findTankEnemies(tankEid: number, { physicalWorld } = GameDI) {
    // Find enemies
    const position = {
        x: RigidBodyState.position.get(tankEid, 0),
        y: RigidBodyState.position.get(tankEid, 1),
    };
    const rotation = RigidBodyState.rotation[tankEid];
    const result = new Set<EntityId>();

    for (let j = 1; j < 5; j++) {
        const radius = 10 ** j;
        physicalWorld.intersectionsWithShape(
            position,
            rotation,
            new Ball(radius),
            (collider: Collider) => {
                const eid = getEntityIdByPhysicalId(collider.handle);
                if (eid !== 0 && tankEid !== eid) {
                    result.add(eid);
                }

                return result.size < TANK_INPUT_TENSOR_MAX_ENEMIES;
            },
            undefined,
            createCollisionGroups(CollisionGroup.TANK_BASE, CollisionGroup.TANK_BASE),
        );
        if (result.size >= TANK_INPUT_TENSOR_MAX_ENEMIES) {
            break;
        }
    }

    return result;
}

export const BULLET_DANGER_SPEED = 100;

export function findTankDangerBullets(tankEid: number, { physicalWorld } = GameDI) {
    const position = {
        x: RigidBodyState.position.get(tankEid, 0),
        y: RigidBodyState.position.get(tankEid, 1),
    };
    const rotation = RigidBodyState.rotation[tankEid];
    const result = new Set<EntityId>();
    const tested = new Set<EntityId>();

    for (let j = 1; j < 5; j++) {
        const radius = 10 ** j;
        physicalWorld.intersectionsWithShape(
            position,
            rotation,
            new Ball(radius),
            (collider: Collider) => {
                const eid = getEntityIdByPhysicalId(collider.handle);

                if (tested.has(eid) || eid === 0 || Player.id[eid] === tankEid) return true;

                tested.add(eid);

                const bulletPosition = RigidBodyState.position.getBatche(eid);
                const bulletVelocity = RigidBodyState.linvel.getBatche(eid);
                const dangerSpeed = hypot(bulletPosition[0], bulletPosition[1]) >= BULLET_DANGER_SPEED;
                const dangerTrajectory = dangerSpeed && hasIntersectionVectorAndCircle(
                    bulletPosition[0],
                    bulletPosition[1],
                    bulletVelocity[0],
                    bulletVelocity[1],
                    position.x,
                    position.y,
                    TANK_APPROXIMATE_COLLISION_RADIUS * 2,
                );

                if (!dangerSpeed || !dangerTrajectory) return true;

                result.add(eid);

                return result.size < TANK_INPUT_TENSOR_MAX_BULLETS;
            },
            undefined,
            createCollisionGroups(CollisionGroup.BULLET, CollisionGroup.BULLET),
        );
        if (result.size >= TANK_INPUT_TENSOR_MAX_BULLETS) {
            break;
        }
    }

    return result;
}

