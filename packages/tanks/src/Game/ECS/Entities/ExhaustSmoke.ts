import { addEntity } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { VFX, VFXType } from '../Components/VFX.ts';
import { Progress } from '../Components/Progress.ts';
import { DestroyByTimeout } from '../Components/Destroy.ts';
import {
    addTransformComponents,
    applyMatrixTranslate,
    applyMatrixScale,
    LocalTransform,
} from '../../../../../renderer/src/ECS/Components/Transform.ts';
import { ZIndex } from '../../consts.ts';
import { RenderDI } from '../../DI/RenderDI.ts';

export const EXHAUST_SMOKE_DURATION = 2_000;

export interface ExhaustSmokeOptions {
    x: number;
    y: number;
    velocityX: number;
    velocityY: number;
    size: number;
}

// Velocity stored per particle (for drift simulation)
const smokeVelocityX = new Float32Array(1024);
const smokeVelocityY = new Float32Array(1024);

export function getSmokeVelocity(eid: number): [number, number] {
    return [smokeVelocityX[eid % 1024], smokeVelocityY[eid % 1024]];
}

export function spawnExhaustSmoke(options: ExhaustSmokeOptions, { world } = GameDI, { enabled } = RenderDI) {
    if (!enabled) return;

    const eid = addEntity(world);

    addTransformComponents(world, eid);
    applyMatrixTranslate(LocalTransform.matrix.getBatch(eid), options.x, options.y, ZIndex.TreadMark);
    applyMatrixScale(LocalTransform.matrix.getBatch(eid), options.size, options.size);

    // Store velocity for drift
    smokeVelocityX[eid % 1024] = options.velocityX;
    smokeVelocityY[eid % 1024] = options.velocityY;

    Progress.addComponent(world, eid, EXHAUST_SMOKE_DURATION);
    VFX.addComponent(world, eid, VFXType.ExhaustSmoke);
    DestroyByTimeout.addComponent(world, eid, EXHAUST_SMOKE_DURATION);

    return eid;
}
