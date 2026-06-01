import { addEntity } from 'bitecs';
import { VFXType } from '../Components/VFX.ts';
import { getFxWorldComponents } from '../createFxWorld.ts';
import { addTransformComponents, applyMatrixTranslate, applyMatrixScale } from '../../../../../renderer/src/ECS/Components/Transform.ts';
import { ZIndex } from '../../consts.ts';
import { RenderDI } from '../../DI/RenderDI.ts';
import { Worlds } from '../../DI/Worlds.ts';

export interface HitFlashOptions {
    x: number;
    y: number;
    size: number;
    duration: number;
}

export function spawnHitFlash(options: HitFlashOptions, { fxWorld } = Worlds, { enabled } = RenderDI) {
    if (!enabled) return;

    const { LocalTransform, VFX, ProgressFx, DestroyByTimeoutFx } = getFxWorldComponents(fxWorld);
    const eid = addEntity(fxWorld);

    addTransformComponents(fxWorld, eid);
    applyMatrixTranslate(LocalTransform.matrix.getBatch(eid), options.x, options.y, ZIndex.HitFlash);
    applyMatrixScale(LocalTransform.matrix.getBatch(eid), options.size, options.size);

    VFX.addComponent(fxWorld, eid, VFXType.HitFlash);
    ProgressFx.addComponent(fxWorld, eid, options.duration);
    DestroyByTimeoutFx.addComponent(fxWorld, eid, options.duration);
}
