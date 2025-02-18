import { DI } from '../../DI';
import { defineQuery } from 'bitecs';
import { Tank } from '../Components/Tank.ts';
import {
    setTankInputTensorEnemy,
    setTankInputTensorSelf,
    TANK_INPUT_TENSOR_MAX_ENEMIES,
    TankInputTensor,
} from '../Components/TankState.ts';
import { getEntityIdByPhysicalId, RigidBodyRef } from '../Components/Physical.ts';
import { length2 } from '../../../../../lib/math.ts';
import { Ball, Collider } from '@dimforge/rapier2d';
import { CollisionGroup, createCollisionGroups } from '../../Physical/createRigid.ts';

export function createTankInputTensorSystem({ world, physicalWorld } = DI) {
    const tanksQuery = defineQuery([Tank, TankInputTensor, RigidBodyRef]);

    const colliderIds = new Float64Array(4);

    return () => {
        const tankEids = tanksQuery(world);

        for (let i = 0; i < tankEids.length; i++) {
            const eid = tankEids[i];
            const pid = RigidBodyRef.id[eid];
            const rb = physicalWorld.getRigidBody(pid);
            const translation = rb.translation();
            const linearVelocity = rb.linvel();
            const rotation = rb.rotation();
            const turretEid = Tank.turretEId[eid];
            const turretPid = RigidBodyRef.id[turretEid];
            const turretRb = physicalWorld.getRigidBody(turretPid);
            const turretRotation = turretRb.rotation();
            const projectileSpeed = Tank.bulletSpeed[eid];

            setTankInputTensorSelf(
                eid,
                translation.x,
                translation.y,
                length2(linearVelocity.x, linearVelocity.y),
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
                    if (pid !== collider.handle) {
                        colliderIds[index++] = collider.handle;
                    }

                    return index < TANK_INPUT_TENSOR_MAX_ENEMIES;
                },
                undefined,
                createCollisionGroups(CollisionGroup.TANK_BASE, CollisionGroup.TANK_BASE),
            );

            console.log('>>', index);
            for (let j = 0; j < TANK_INPUT_TENSOR_MAX_ENEMIES; j++) {
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
                    eid,
                    j,
                    translation.x,
                    translation.y,
                    length2(linearVelocity.x, linearVelocity.y),
                    rotation,
                    turretRotation,
                );
            }
        }
    };
}
