import { ActiveEvents, ColliderDesc } from '@dimforge/rapier2d';
import { DI } from '../DI';
import { BodyOptions, createBody } from './createBody.ts';
import { ActiveCollisionTypes } from '@dimforge/rapier2d/src/geometry/collider.ts';
import { RigidBodyRef } from '../ECS/Components/Physical.ts';
import { removeEntity } from 'bitecs';

export enum CollisionGroup {
    ALL = 0xFFFF,
    WALL = 0b0001,   // 1
    TANK = 0b0010,   // 2
    BULLET = 0b0100, // 4
}

type CommonRigidOptions = BodyOptions & {
    mass?: number,
    belongsCollisionGroup?: 0 | CollisionGroup,
    interactsCollisionGroup?: 0 | CollisionGroup,
    belongsSolverGroup?: 0 | CollisionGroup,
    interactsSolverGroup?: 0 | CollisionGroup,
    collisionEvent?: ActiveEvents
    activeCollisionTypes?: ActiveCollisionTypes
}

function prepareColliderDesc(shape: ColliderDesc, o: CommonRigidOptions): ColliderDesc {
    return shape.setDensity(o.mass ?? 0)
        .setCollisionGroups(((o.belongsCollisionGroup ?? CollisionGroup.ALL) << 16) | (o.interactsCollisionGroup ?? CollisionGroup.ALL))
        .setSolverGroups(((o.belongsSolverGroup ?? CollisionGroup.ALL) << 16) | (o.interactsSolverGroup ?? CollisionGroup.ALL))
        .setActiveEvents(o.collisionEvent ?? ActiveEvents.NONE)
        .setActiveCollisionTypes(o.activeCollisionTypes ?? ActiveCollisionTypes.DEFAULT);
}

export function createRigidRectangle(
    o: CommonRigidOptions & {
        width: number,
        height: number,
        rotation: number,
    },
    { physicalWorld } = DI,
) {
    const body = createBody(o);
    const colliderDesc = prepareColliderDesc(ColliderDesc.cuboid(o.width / 2, o.height / 2), o);
    physicalWorld.createCollider(colliderDesc, body);

    return body.handle;
}

export function createRigidCircle(
    o: CommonRigidOptions & {
        x: number,
        y: number,
        radius: number,
    },
    { physicalWorld } = DI,
) {
    const body = createBody(o);
    const colliderDesc = prepareColliderDesc(ColliderDesc.ball(o.radius / 2), o);
    physicalWorld.createCollider(colliderDesc, body);

    return body.handle;
}

export function removeRigidEntity(eid: number, { world, physicalWorld } = DI) {
    const pid = RigidBodyRef.id[eid];
    const body = physicalWorld.getRigidBody(pid);
    physicalWorld.removeRigidBody(body);
    removeEntity(world, eid);
}