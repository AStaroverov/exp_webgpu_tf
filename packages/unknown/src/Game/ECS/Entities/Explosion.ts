import { addEntity } from 'bitecs';
import { VFXType } from '../Components/VFX.ts';
import { getFxWorldComponents } from '../createFxWorld.ts';
import { addTransformComponents, applyMatrixTranslate, applyMatrixScale } from '../../../../../renderer/src/ECS/Components/Transform.ts';
import { ZIndex } from '../../consts.ts';
import { RenderDI } from '../../DI/RenderDI.ts';
import { Worlds } from '../../DI/Worlds.ts';

export interface ExplosionOptions {
    x: number;
    y: number;
    size: number;
    duration: number;
}

export function spawnExplosion(options: ExplosionOptions, { fxWorld } = Worlds, { enabled } = RenderDI) {
    if (!enabled) return;

    const { LocalTransform, VFX, ProgressFx, DestroyByTimeoutFx } = getFxWorldComponents(fxWorld);
    const eid = addEntity(fxWorld);

    addTransformComponents(fxWorld, eid);
    applyMatrixTranslate(LocalTransform.matrix.getBatch(eid), options.x, options.y, ZIndex.Explosion);
    applyMatrixScale(LocalTransform.matrix.getBatch(eid), options.size, options.size);

    VFX.addComponent(fxWorld, eid, VFXType.Explosion);
    ProgressFx.addComponent(fxWorld, eid, options.duration);
    DestroyByTimeoutFx.addComponent(fxWorld, eid, options.duration);
}
