import { ActiveEvents, ColliderDesc, RigidBodyDesc } from '@dimforge/rapier2d';
import { RigidBodyType } from '@dimforge/rapier2d/src/dynamics/rigid_body.ts';
import { PhysicalWorld } from '../index.ts';

type CommonPhysicalOptions = {
    x: number,
    y: number,
    bodyType: RigidBodyType,
    gravityScale: number,
    density: number,
    collisionEvent: ActiveEvents
}

export function createRectangleCollider(
    physicalWorld: PhysicalWorld,
    o: CommonPhysicalOptions & {
        width: number,
        height: number,
        rotation: number,
    },
) {
    let bodyDesc = (new RigidBodyDesc(o.bodyType))
        .setTranslation(o.x, o.y)
        .setRotation(o.rotation)
        .setGravityScale(o.gravityScale)
        .setCcdEnabled(true);
    let body = physicalWorld.createRigidBody(bodyDesc);
    let colliderDesc = ColliderDesc.cuboid(o.width / 2, o.height / 2)
        .setDensity(o.density)
        .setActiveEvents(o.collisionEvent);
    physicalWorld.createCollider(colliderDesc, body);

    return body.handle;
}

export function createCircleCollider(
    physicalWorld: PhysicalWorld,
    o: CommonPhysicalOptions & {
        x: number,
        y: number,
        radius: number,
        bodyType: RigidBodyType,
        density: number,
    },
) {
    let bodyDesc = (new RigidBodyDesc(o.bodyType))
        .setTranslation(o.x, o.y)
        .setGravityScale(o.gravityScale)
        .setCcdEnabled(true);

    let body = physicalWorld.createRigidBody(bodyDesc);
    let colliderDesc = ColliderDesc.ball(o.radius / 2)
        .setDensity(o.density)
        .setActiveEvents(o.collisionEvent);
    physicalWorld.createCollider(colliderDesc, body);

    return body.handle;
}