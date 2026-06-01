import { addEntity } from 'bitecs';
import { VFXType } from '../Components/VFX.ts';
import { getFxWorldComponents } from '../createFxWorld.ts';
import {
    addTransformComponents,
    applyMatrixTranslate,
    applyMatrixScale,
} from '../../../../../renderer/src/ECS/Components/Transform.ts';
import { ZIndex } from '../../consts.ts';
import { RenderDI } from '../../DI/RenderDI.ts';
import { Worlds } from '../../DI/Worlds.ts';

export const EXHAUST_SMOKE_DURATION = 2_000;

export interface ExhaustSmokeOptions {
    x: number;
    y: number;
    velocityX: number;
    velocityY: number;
    size: number;
}

export function spawnExhaustSmoke(options: ExhaustSmokeOptions, { fxWorld } = Worlds, { enabled } = RenderDI) {
    if (!enabled) return;

    const { LocalTransform, ProgressFx, VFX, DestroyByTimeoutFx } = getFxWorldComponents(fxWorld);
    const eid = addEntity(fxWorld);

    addTransformComponents(fxWorld, eid);
    applyMatrixTranslate(LocalTransform.matrix.getBatch(eid), options.x, options.y, ZIndex.TreadMark);
    applyMatrixScale(LocalTransform.matrix.getBatch(eid), options.size, options.size);

    ProgressFx.addComponent(fxWorld, eid, EXHAUST_SMOKE_DURATION);
    VFX.addComponent(fxWorld, eid, VFXType.ExhaustSmoke);
    DestroyByTimeoutFx.addComponent(fxWorld, eid, EXHAUST_SMOKE_DURATION);

    return eid;
}
