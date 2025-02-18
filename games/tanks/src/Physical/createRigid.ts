import { ActiveEvents, ColliderDesc } from '@dimforge/rapier2d';
import { DI } from '../DI';
import { BodyOptions, createBody } from './createBody.ts';
import { ActiveCollisionTypes } from '@dimforge/rapier2d/src/geometry/collider.ts';
import { RigidBodyRef } from '../ECS/Components/Physical.ts';

export enum CollisionGroup {
    ALL = 0xFFFF,
    WALL = 0b0000001,
    BULLET = 0b0000010,
    TANK_BASE = 0b0000100,
    TANK_BODY_PARTS = 0b0001000,
    TANK_TURRET_PARTS = 0b0100000,
    TANK_GUN_PARTS = 0b1000000,
    TANK_PARTS = CollisionGroup.TANK_BODY_PARTS | CollisionGroup.TANK_TURRET_PARTS | CollisionGroup.TANK_GUN_PARTS,
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

export function createCollisionGroups(belongs: 0 | CollisionGroup, interacts: 0 | CollisionGroup) {
    return (belongs << 16) | interacts;
}

function prepareColliderDesc(shape: ColliderDesc, o: CommonRigidOptions): ColliderDesc {
    return shape
        .setDensity(o.density ?? 0)
        .setCollisionGroups(
            createCollisionGroups(
                o.belongsCollisionGroup ?? CollisionGroup.ALL,
                o.interactsCollisionGroup ?? CollisionGroup.ALL,
            ),
        )
        .setSolverGroups(
            createCollisionGroups(
                o.belongsSolverGroup ?? CollisionGroup.ALL,
                o.interactsSolverGroup ?? CollisionGroup.ALL,
            ),
        )
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