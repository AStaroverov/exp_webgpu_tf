import { ActiveEvents, ColliderDesc } from '@dimforge/rapier2d';
import { DI } from '../DI';
import { BodyOptions, createBody } from './createBody.ts';
import { ActiveCollisionTypes } from '@dimforge/rapier2d/src/geometry/collider.ts';

type CommonRigidOptions = BodyOptions & {
    mass?: number,
    belongsCollisionGroup?: number,
    interactsCollisionGroup?: number,
    belongsSolverGroup?: number,
    interactsSolverGroup?: number,

    activeCollisionTypes?: ActiveCollisionTypes
}

export type RigidRectangleOptions = CommonRigidOptions & {
    width: number,
    height: number,
    rotation: number,
}

export enum CollisionGroup {
    ALL = 0xFFFF,
    WALL = 0b0001,   // 1
    TANK = 0b0010,   // 2
    BULLET = 0b0100, // 4
}

function prepareColliderDesc(shape: ColliderDesc, o: CommonRigidOptions): ColliderDesc {
    return shape.setDensity(o.mass ?? 0)
        .setCollisionGroups(((o.belongsCollisionGroup ?? CollisionGroup.ALL) << 16) | (o.interactsCollisionGroup ?? CollisionGroup.ALL))
        .setSolverGroups(((o.belongsSolverGroup ?? CollisionGroup.ALL) << 16) | (o.interactsSolverGroup ?? CollisionGroup.ALL))
        .setActiveEvents(o.collisionEvent ?? ActiveEvents.NONE)
        .setActiveCollisionTypes(o.activeCollisionTypes ?? ActiveCollisionTypes.DEFAULT);
}

export function createRigidRectangle(
    o: RigidRectangleOptions,
    { physicalWorld } = DI,
) {
    const body = createBody(o);
    const colliderDesc = prepareColliderDesc(ColliderDesc.cuboid(o.width / 2, o.height / 2), o);
    physicalWorld.createCollider(colliderDesc, body);

    return body.handle;
}

export type RigidCircleOptions = CommonRigidOptions & {
    x: number,
    y: number,
    radius: number,
}

export function createRigidCircle(
    o: RigidCircleOptions,
    { physicalWorld } = DI,
) {
    const body = createBody(o);
    const colliderDesc = prepareColliderDesc(ColliderDesc.ball(o.radius / 2), o);
    physicalWorld.createCollider(colliderDesc, body);

    return body.handle;
}