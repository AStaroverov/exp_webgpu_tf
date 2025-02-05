import { ActiveEvents, ColliderDesc, RigidBodyDesc } from '@dimforge/rapier2d';
import { RigidBodyType } from '@dimforge/rapier2d/src/dynamics/rigid_body.ts';
import { PhysicalWorld } from '../index.ts';

type CommonPhysicalOptions = {
    x: number,
    y: number,
    bodyType: RigidBodyType,
    gravityScale: number,
    mass: number,
    collisionEvent: ActiveEvents
}

export type RectangleColliderOptions = CommonPhysicalOptions & {
    width: number,
    height: number,
    rotation: number,
}

export function createRectangleCollider(
    physicalWorld: PhysicalWorld,
    o: RectangleColliderOptions,
) {
    let bodyDesc = (new RigidBodyDesc(o.bodyType))
        .setTranslation(o.x, o.y)
        .setRotation(o.rotation)
        .setGravityScale(o.gravityScale)
        .setCcdEnabled(true);
    let body = physicalWorld.createRigidBody(bodyDesc);
    let colliderDesc = ColliderDesc.cuboid(o.width / 2, o.height / 2)
        .setMass(o.mass)
        .setActiveEvents(o.collisionEvent);
    physicalWorld.createCollider(colliderDesc, body);

    return body.handle;
}

export type CircleColliderOptions = CommonPhysicalOptions & {
    x: number,
    y: number,
    radius: number,
    bodyType: RigidBodyType,
    mass: number,
}

export function createCircleCollider(
    physicalWorld: PhysicalWorld,
    o: CircleColliderOptions,
) {
    let bodyDesc = (new RigidBodyDesc(o.bodyType))
        .setTranslation(o.x, o.y)
        .setGravityScale(o.gravityScale)
        .setCcdEnabled(true);

    let body = physicalWorld.createRigidBody(bodyDesc);
    let colliderDesc = ColliderDesc.ball(o.radius / 2)
        .setMass(o.mass)
        .setActiveEvents(o.collisionEvent);
    physicalWorld.createCollider(colliderDesc, body);

    return body.handle;
}