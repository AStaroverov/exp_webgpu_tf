import { DI } from '../../DI';
import { getTankHealth, Tank } from '../Components/Tank.ts';
import {
    setTankInputTensorSelf,
    TANK_INPUT_TENSOR_MAX_BULLETS,
    TANK_INPUT_TENSOR_MAX_ENEMIES,
    TankInputTensor,
} from '../Components/TankState.ts';
import { getEntityIdByPhysicalId, RigidBodyRef } from '../Components/Physical.ts';
import { hypot, max } from '../../../../../lib/math.ts';
import { Ball, Collider } from '@dimforge/rapier2d';
import { CollisionGroup, createCollisionGroups } from '../../Physical/createRigid.ts';
import { query } from 'bitecs';
import { Player } from '../Components/Player.ts';

export function createTankInputTensorSystem({ world, physicalWorld } = DI) {
    const colliderIds = new Float64Array(max(TANK_INPUT_TENSOR_MAX_ENEMIES, TANK_INPUT_TENSOR_MAX_BULLETS));

    return () => {
        const tankEids = query(world, [Tank, TankInputTensor, RigidBodyRef]);

        TankInputTensor.resetEnemiesCoords();
        TankInputTensor.resetBulletsCoords();

        for (let i = 0; i < tankEids.length; i++) {
            const tankEid = tankEids[i];
            const tankPid = RigidBodyRef.id[tankEid];
            const playerId = Player.id[tankEid];
            const rb = physicalWorld.getRigidBody(tankPid);
            const health = getTankHealth(tankEid);
            const translation = rb.translation();
            const linearVelocity = rb.linvel();
            const rotation = rb.rotation();
            const turretEid = Tank.turretEId[tankEid];
            const turretPid = RigidBodyRef.id[turretEid];
            const turretRb = physicalWorld.getRigidBody(turretPid);
            const turretRotation = turretRb.rotation();
            const projectileSpeed = Tank.bulletSpeed[tankEid];

            setTankInputTensorSelf(
                tankEid,
                health,
                translation.x,
                translation.y,
                hypot(linearVelocity.x, linearVelocity.y),
                rotation,
                turretRotation,
                projectileSpeed,
            );

            // Find enemies
            let enemyIndex = 0;

            for (let j = 1; j < 5; j++) {
                const radius = 10 ** j;
                physicalWorld.intersectionsWithShape(
                    translation,
                    rotation,
                    new Ball(radius),
                    (collider: Collider) => {
                        if (tankPid !== collider.handle) {
                            colliderIds[enemyIndex++] = collider.handle;
                        }

                        return enemyIndex < TANK_INPUT_TENSOR_MAX_ENEMIES;
                    },
                    undefined,
                    createCollisionGroups(CollisionGroup.TANK_BASE, CollisionGroup.TANK_BASE),
                );
                if (enemyIndex >= TANK_INPUT_TENSOR_MAX_ENEMIES) {
                    break;
                }
            }

            for (let j = 0; j < enemyIndex; j++) {
                const pid = colliderIds[j];
                const rb = physicalWorld.getRigidBody(pid);

                rb && TankInputTensor.setEnemiesData(
                    tankEid,
                    j,
                    getEntityIdByPhysicalId(pid),
                    rb.translation(),
                    rb.linvel(),
                );
            }

            // Find bullets
            let bulletIndex = 0;

            for (let j = 1; j < 5; j++) {
                const radius = 10 ** j;
                physicalWorld.intersectionsWithShape(
                    translation,
                    rotation,
                    new Ball(radius),
                    (collider: Collider) => {
                        const eid = getEntityIdByPhysicalId(collider.handle);
                        const vel = collider.parent()?.linvel();

                        if (Player.id[eid] === playerId || vel == null || hypot(vel.x, vel.y) < 100) {
                            return true;
                        }

                        colliderIds[bulletIndex++] = collider.handle;

                        return bulletIndex < TANK_INPUT_TENSOR_MAX_BULLETS;
                    },
                    undefined,
                    createCollisionGroups(CollisionGroup.BULLET, CollisionGroup.BULLET),
                );
                if (bulletIndex >= TANK_INPUT_TENSOR_MAX_BULLETS) {
                    break;
                }
            }

            for (let j = 0; j < bulletIndex; j++) {
                const pid = colliderIds[j];
                const rb = physicalWorld.getRigidBody(pid);

                rb && TankInputTensor.setBulletsData(
                    tankEid,
                    j,
                    getEntityIdByPhysicalId(pid),
                    rb.translation(),
                    rb.linvel(),
                );
            }
        }
    };
}
