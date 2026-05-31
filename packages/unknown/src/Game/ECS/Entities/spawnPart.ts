import { addEntity } from 'bitecs';
import {
    addTransformComponents,
    applyMatrixTranslate,
} from '../../../../../renderer/src/ECS/Components/Transform.ts';
import { createRectangle, createCircle } from '../../../../../renderer/src/ECS/Entities/Shapes.ts';
import { getRenderComponents } from '../../../../../renderer/src/ECS/world.ts';
import { createRigidRectangle, createRigidCircle } from '../../Physical/createRigid.ts';
import { PhysicalWorld } from '../../Physical/initPhysicalWorld.ts';
import { getPhysicsWorldComponents, PhysicsWorld } from '../createPhysicsWorld.ts';
import { RenderGameWorld } from '../createRenderWorld.ts';
import { BridgeDI } from '../../DI/BridgeDI.ts';

export type SpawnCtx = {
    physicsWorld: PhysicsWorld;
    renderWorld: RenderGameWorld;
    physicalWorld: PhysicalWorld;
};

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

function linkAtom(physEid: number, renderEid: number, physicalId: number): SpawnResult {
    BridgeDI.link('mirror', physEid, renderEid);
    BridgeDI.registerPhysicalId(physicalId, physEid);
    return [physEid, renderEid, physicalId];
}

// ---- visible part / bullet: mirror carries Shape/Color/Roundness (was createRectangleRR) ----
export function spawnRectanglePart(
    ctx: SpawnCtx,
    options: Parameters<typeof createRectangle>[1] & Parameters<typeof createRigidRectangle>[1],
): SpawnResult {
    const physicalId = createRigidRectangle(ctx.physicalWorld, options);
    const physEid = addPhysicsAtom(ctx.physicsWorld, physicalId);
    const renderEid = createRectangle(ctx.renderWorld, options);
    return linkAtom(physEid, renderEid, physicalId);
}

export function spawnCirclePart(
    ctx: SpawnCtx,
    options: Parameters<typeof createCircle>[1] & Parameters<typeof createRigidCircle>[1],
): SpawnResult {
    const physicalId = createRigidCircle(ctx.physicalWorld, options);
    const physEid = addPhysicsAtom(ctx.physicsWorld, physicalId);
    const renderEid = createCircle(ctx.renderWorld, options);
    return linkAtom(physEid, renderEid, physicalId);
}

// ---- carrier: invisible body, mirror is transform-only (was createRectangleRigidGroup) ----
export function spawnRectangleCarrier(
    ctx: SpawnCtx,
    options: Parameters<typeof createRigidRectangle>[1] & { x: number; y: number },
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
    ctx: SpawnCtx,
    options: Parameters<typeof createRigidCircle>[1] & { x: number; y: number },
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
