import { DI } from '../../DI';
import { getTankHealth, Tank } from '../Components/Tank.ts';
import {
    resetTankInputTensorEnemy,
    setTankInputTensorEnemy,
    setTankInputTensorSelf,
    TANK_INPUT_TENSOR_MAX_ENEMIES,
    TankInputTensor,
} from '../Components/TankState.ts';
import { getEntityIdByPhysicalId, RigidBodyRef } from '../Components/Physical.ts';
import { hypot } from '../../../../../lib/math.ts';
import { Ball, Collider } from '@dimforge/rapier2d';
import { CollisionGroup, createCollisionGroups } from '../../Physical/createRigid.ts';
import { query } from 'bitecs';

export function createTankInputTensorSystem({ world, physicalWorld } = DI) {
    const colliderIds = new Float64Array(4);

    return () => {
        const tankEids = query(world, [Tank, TankInputTensor, RigidBodyRef]);

        for (let i = 0; i < tankEids.length; i++) {
            const tankEid = tankEids[i];
            const tankPid = RigidBodyRef.id[tankEid];
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

            let index = 0;
            physicalWorld.intersectionsWithShape(
                translation,
                rotation,
                new Ball(10000),
                (collider: Collider) => {
                    if (tankPid !== collider.handle) {
                        colliderIds[index++] = collider.handle;
                    }

                    return index < TANK_INPUT_TENSOR_MAX_ENEMIES;
                },
                undefined,
                createCollisionGroups(CollisionGroup.TANK_BASE, CollisionGroup.TANK_BASE),
            );

            for (let j = 0; j < index; j++) {
                const pid = colliderIds[j];
                const eid = getEntityIdByPhysicalId(pid);
                const rb = physicalWorld.getRigidBody(pid);
                const translation = rb.translation();
                const rotation = rb.rotation();
                const linearVelocity = rb.linvel();
                const turretEid = Tank.turretEId[eid];
                const turretPid = RigidBodyRef.id[turretEid];
                const turretRb = physicalWorld.getRigidBody(turretPid);
                const turretRotation = turretRb.rotation();

                setTankInputTensorEnemy(
                    tankEid,
                    j,
                    translation.x,
                    translation.y,
                    hypot(linearVelocity.x, linearVelocity.y),
                    rotation,
                    turretRotation,
                );
            }

            for (let j = index; j < TANK_INPUT_TENSOR_MAX_ENEMIES; j++) {
                resetTankInputTensorEnemy(tankEid, j);
            }
        }
    };
}
