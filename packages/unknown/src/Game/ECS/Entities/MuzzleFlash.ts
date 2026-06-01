import { addEntity } from 'bitecs';
import { VFXType } from '../Components/VFX.ts';
import { getFxWorldComponents } from '../createFxWorld.ts';
import { addTransformComponents, applyMatrixTranslate, applyMatrixRotateZ, applyMatrixScale } from '../../../../../renderer/src/ECS/Components/Transform.ts';
import { ZIndex } from '../../consts.ts';
import { RenderDI } from '../../DI/RenderDI.ts';
import { Worlds } from '../../DI/Worlds.ts';

export interface MuzzleFlashOptions {
    x: number;
    y: number;
    size: number;
    duration: number;
    rotation?: number;
}

export function spawnMuzzleFlash(options: MuzzleFlashOptions, { fxWorld } = Worlds, { enabled } = RenderDI) {
    if (!enabled) return;

    const { LocalTransform, VFX, ProgressFx, DestroyByTimeoutFx } = getFxWorldComponents(fxWorld);
    const eid = addEntity(fxWorld);

    addTransformComponents(fxWorld, eid);
    applyMatrixTranslate(LocalTransform.matrix.getBatch(eid), options.x, options.y, ZIndex.MuzzleFlash);
    if (options.rotation) {
        applyMatrixRotateZ(LocalTransform.matrix.getBatch(eid), options.rotation);
    }
    applyMatrixScale(LocalTransform.matrix.getBatch(eid), options.size, options.size);

    VFX.addComponent(fxWorld, eid, VFXType.MuzzleFlash);
    ProgressFx.addComponent(fxWorld, eid, options.duration);
    DestroyByTimeoutFx.addComponent(fxWorld, eid, options.duration);
}
