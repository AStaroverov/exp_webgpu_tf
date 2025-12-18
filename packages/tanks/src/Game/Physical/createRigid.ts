import { ActiveCollisionTypes, ActiveEvents, ColliderDesc } from '@dimforge/rapier2d-simd';
import { GameDI } from '../DI/GameDI.ts';
import { BodyOptions, createBody } from './createBody.ts';
import { RigidBodyRef } from '../ECS/Components/Physical.ts';
import { PhysicalWorld } from './initPhysicalWorld.ts';

export enum CollisionGroup {
    NONE = 0,
    ALL = 0xFFFF,
    WALL = 0b00000001,
    BULLET = 0b00000010,
    TANK_BASE = 0b00000100,
    TANK_HULL_PARTS = 0b00001000,
    TANK_TURRET_HEAD_PARTS = 0b00100000,
    TANK_TURRET_GUN_PARTS = 0b01000000,
    SHIELD = 0b10000000, // Shield parts - only interact with bullets, not with each other
    TANK_PARTS = CollisionGroup.TANK_HULL_PARTS | CollisionGroup.TANK_TURRET_HEAD_PARTS | CollisionGroup.TANK_TURRET_GUN_PARTS,
}

type CommonRigidOptions = BodyOptions & {
    enabled?: boolean,
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
        .setEnabled(o.enabled ?? true)
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
    },
    { physicalWorld }: { physicalWorld: PhysicalWorld } = GameDI,
) {
    const body = createBody(o, { physicalWorld });
    const colliderDesc = prepareColliderDesc(ColliderDesc.cuboid(o.width / 2, o.height / 2), o);
    physicalWorld.createCollider(colliderDesc, body);

    return body.handle;
}

export function createRigidCircle(
    o: CommonRigidOptions & {
        radius: number,
    },
    { physicalWorld }: { physicalWorld: PhysicalWorld } = GameDI,
) {
    const body = createBody(o);
    const colliderDesc = prepareColliderDesc(ColliderDesc.ball(o.radius / 2), o);
    physicalWorld.createCollider(colliderDesc, body);

    return body.handle;
}

export function removeRigidShape(eid: number, { physicalWorld } = GameDI) {
    const pid = RigidBodyRef.id[eid];
    const body = physicalWorld.getRigidBody(pid);
    body && physicalWorld.removeRigidBody(body);
}