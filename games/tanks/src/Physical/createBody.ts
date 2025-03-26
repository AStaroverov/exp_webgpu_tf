import { RigidBody, RigidBodyDesc, RigidBodyType } from '@dimforge/rapier2d';
import { GameDI } from '../DI/GameDI.ts';

export type BodyOptions = {
    x: number,
    y: number,
    speedX?: number,
    speedY?: number,
    rotation?: number,
    bodyType?: RigidBodyType,
    gravityScale?: number,
    linearDamping?: number,
    angularDamping?: number,
    additionalMass?: number,
    continuousCollisionDetection?: boolean,
}

export function createBody(
    o: BodyOptions,
    { physicalWorld } = GameDI,
): RigidBody {
    let bodyDesc = (new RigidBodyDesc(o.bodyType ?? RigidBodyType.Dynamic))
        .setAdditionalMass(o.additionalMass ?? 0)
        .setTranslation(o.x, o.y)
        .setRotation(o.rotation ?? 0)
        .setGravityScale(o.gravityScale ?? 0)
        .setLinvel(o.speedX ?? 0, o.speedY ?? 0)
        .setLinearDamping(o.linearDamping ?? 1)
        .setAngularDamping(o.angularDamping ?? 1)
        .setCcdEnabled(o.continuousCollisionDetection ?? false);
    return physicalWorld.createRigidBody(bodyDesc);
}

export function removeBody(body: RigidBody, { physicalWorld } = GameDI) {
    physicalWorld.removeRigidBody(body);
}