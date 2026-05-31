import { addEntity } from 'bitecs';
import { VFXType } from '../Components/VFX.ts';
import { getRenderWorldComponents, RenderGameWorld } from '../createRenderWorld.ts';
import { addTransformComponents, applyMatrixTranslate, applyMatrixScale, LocalTransform } from '../../../../../renderer/src/ECS/Components/Transform.ts';
import { ZIndex } from '../../consts.ts';
import { RenderDI } from '../../DI/RenderDI.ts';

export interface ExplosionOptions {
    x: number;
    y: number;
    size: number;
    duration: number;
}

export function spawnExplosion(world: RenderGameWorld, options: ExplosionOptions, { enabled } = RenderDI) {
    if (!enabled) return;

    const { VFX, ProgressFx, DestroyByTimeoutFx } = getRenderWorldComponents(world);
    const eid = addEntity(world);

    addTransformComponents(world, eid);
    applyMatrixTranslate(LocalTransform.matrix.getBatch(eid), options.x, options.y, ZIndex.Explosion);
    applyMatrixScale(LocalTransform.matrix.getBatch(eid), options.size, options.size);

    VFX.addComponent(world, eid, VFXType.Explosion);
    ProgressFx.addComponent(world, eid, options.duration);
    DestroyByTimeoutFx.addComponent(world, eid, options.duration);
}
