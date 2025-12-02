import { addEntity, EntityId } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { HitFlash } from '../Components/HitFlash.ts';
import { Progress } from '../Components/Progress.ts';
import { DestroyByTimeout } from '../Components/Destroy.ts';
import { addTransformComponents, applyMatrixTranslate, applyMatrixScale, LocalTransform } from '../../../../../renderer/src/ECS/Components/Transform.ts';
import { ZIndex } from '../../consts.ts';

export interface HitFlashOptions {
    x: number;
    y: number;
    size: number;
    duration: number;
}

export function createHitFlash(options: HitFlashOptions, { world } = GameDI): EntityId {
    const eid = addEntity(world);

    addTransformComponents(world, eid);
    applyMatrixTranslate(LocalTransform.matrix.getBatch(eid), options.x, options.y, ZIndex.HitFlash);
    applyMatrixScale(LocalTransform.matrix.getBatch(eid), options.size, options.size);

    HitFlash.addComponent(world, eid);
    Progress.addComponent(world, eid, options.duration);
    DestroyByTimeout.addComponent(world, eid, options.duration);

    return eid;
}
