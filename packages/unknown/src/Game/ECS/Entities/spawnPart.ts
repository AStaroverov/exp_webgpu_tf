import { addEntity } from 'bitecs';
import {
    addTransformComponents,
    applyMatrixTranslate,
} from '../../../../../renderer/src/ECS/Components/Transform.ts';
import { createRectangle, createCircle } from '../../../../../renderer/src/ECS/Entities/Shapes.ts';
import { getRenderComponents } from '../../../../../renderer/src/ECS/world.ts';
import { createRigidRectangle, createRigidCircle } from '../../Physical/createRigid.ts';
import { getPhysicsWorldComponents, PhysicsWorld } from '../createPhysicsWorld.ts';
import { getRenderWorldComponents } from '../createRenderWorld.ts';
import { physicsByBody } from '../../DI/physicsByBody.ts';
import { Worlds } from '../../DI/Worlds.ts';


export type SpawnResult = [physEid: number, renderEid: number, physicalId: number];

function addPhysicsAtom(physicsWorld: PhysicsWorld, physicalId: number): number {
    const P = getPhysicsWorldComponents(physicsWorld);
    const physEid = addEntity(physicsWorld);
    P.RigidBodyRef.addComponent(physicsWorld, physEid, physicalId);
    P.RigidBodyState.addComponent(physicsWorld, physEid);
    P.Impulse.addComponent(physicsWorld, physEid);
    P.TorqueImpulse.addComponent(physicsWorld, physEid);
    return physEid;
}

// The render references its physics body downward (render -> physics). The atom no
// longer references its render; the owning brain node points at the render/physics.
function linkAtom(physEid: number, renderEid: number, physicalId: number, { renderWorld } = Worlds): SpawnResult {
    getRenderWorldComponents(renderWorld).PhysicsRef.set(renderWorld, renderEid, physEid);
    physicsByBody.set(physicalId, physEid);
    return [physEid, renderEid, physicalId];
}

// ---- visible part / bullet: mirror carries Shape/Color/Roundness (was createRectangleRR) ----
export function spawnRectanglePart(
    options: Parameters<typeof createRectangle>[1] & Parameters<typeof createRigidRectangle>[1],
    ctx = Worlds,
): SpawnResult {
    const physicalId = createRigidRectangle(ctx.physicalWorld, options);
    const physEid = addPhysicsAtom(ctx.physicsWorld, physicalId);
    const renderEid = createRectangle(ctx.renderWorld, options);
    return linkAtom(physEid, renderEid, physicalId);
}

export function spawnCirclePart(
    options: Parameters<typeof createCircle>[1] & Parameters<typeof createRigidCircle>[1],
    ctx = Worlds,
): SpawnResult {
    const physicalId = createRigidCircle(ctx.physicalWorld, options);
    const physEid = addPhysicsAtom(ctx.physicsWorld, physicalId);
    const renderEid = createCircle(ctx.renderWorld, options);
    return linkAtom(physEid, renderEid, physicalId);
}

// ---- carrier: invisible body, mirror is transform-only (was createRectangleRigidGroup) ----
export function spawnRectangleCarrier(
    options: Parameters<typeof createRigidRectangle>[1] & { x: number; y: number },
    ctx = Worlds,
): SpawnResult {
    const physicalId = createRigidRectangle(ctx.physicalWorld, options);
    const physEid = addPhysicsAtom(ctx.physicsWorld, physicalId);
    const renderEid = addEntity(ctx.renderWorld);
    addTransformComponents(ctx.renderWorld, renderEid);
    applyMatrixTranslate(
        getRenderComponents(ctx.renderWorld).LocalTransform.matrix.getBatch(renderEid),
        options.x,
        options.y,
        0,
    );
    return linkAtom(physEid, renderEid, physicalId);
}

export function spawnCircleCarrier(
    options: Parameters<typeof createRigidCircle>[1] & { x: number; y: number },
    ctx = Worlds,
): SpawnResult {
    const physicalId = createRigidCircle(ctx.physicalWorld, options);
    const physEid = addPhysicsAtom(ctx.physicsWorld, physicalId);
    const renderEid = addEntity(ctx.renderWorld);
    addTransformComponents(ctx.renderWorld, renderEid);
    applyMatrixTranslate(
        getRenderComponents(ctx.renderWorld).LocalTransform.matrix.getBatch(renderEid),
        options.x,
        options.y,
        0,
    );
    return linkAtom(physEid, renderEid, physicalId);
}
