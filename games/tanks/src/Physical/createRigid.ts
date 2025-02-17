import { ActiveEvents, ColliderDesc } from '@dimforge/rapier2d';
import { DI } from '../DI';
import { BodyOptions, createBody } from './createBody.ts';
import { ActiveCollisionTypes } from '@dimforge/rapier2d/src/geometry/collider.ts';
import { RigidBodyRef } from '../ECS/Components/Physical.ts';

export enum CollisionGroup {
    ALL = 0xFFFF,
    WALL = 0b000001,
    BULLET = 0b000010,
    TANK_1 = 0b000100,
    TANK_2 = 0b001000,
    TANK_3 = 0b010000,
    TANK = CollisionGroup.TANK_1 | CollisionGroup.TANK_2 | CollisionGroup.TANK_3,
}


type CommonRigidOptions = BodyOptions & {
    density?: number,
    belongsCollisionGroup?: 0 | CollisionGroup,
    interactsCollisionGroup?: 0 | CollisionGroup,
    belongsSolverGroup?: 0 | CollisionGroup,
    interactsSolverGroup?: 0 | CollisionGroup,
    collisionEvent?: ActiveEvents
    activeCollisionTypes?: ActiveCollisionTypes
}

function prepareColliderDesc(shape: ColliderDesc, o: CommonRigidOptions): ColliderDesc {
    return shape
        .setDensity(o.density ?? 0)
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

export function removeRigidShape(eid: number, { physicalWorld } = DI) {
    const pid = RigidBodyRef.id[eid];
    const body = physicalWorld.getRigidBody(pid);
    body && physicalWorld.removeRigidBody(body);
}