import { addEntity, EntityId } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { MuzzleFlash } from '../Components/MuzzleFlash.ts';
import { Progress } from '../Components/Progress.ts';
import { DestroyByTimeout } from '../Components/Destroy.ts';
import { addTransformComponents, applyMatrixTranslate, applyMatrixRotateZ, applyMatrixScale, LocalTransform } from '../../../../../renderer/src/ECS/Components/Transform.ts';
import { ZIndex } from '../../consts.ts';

export interface MuzzleFlashOptions {
    x: number;
    y: number;
    size: number;
    duration: number;
    rotation?: number;
}

export function createMuzzleFlash(options: MuzzleFlashOptions, { world } = GameDI): EntityId {
    const eid = addEntity(world);

    addTransformComponents(world, eid);
    applyMatrixTranslate(LocalTransform.matrix.getBatch(eid), options.x, options.y, ZIndex.MuzzleFlash);
    if (options.rotation) {
        applyMatrixRotateZ(LocalTransform.matrix.getBatch(eid), options.rotation);
    }
    applyMatrixScale(LocalTransform.matrix.getBatch(eid), options.size, options.size);

    MuzzleFlash.addComponent(world, eid);
    Progress.addComponent(world, eid, options.duration);
    DestroyByTimeout.addComponent(world, eid, options.duration);

    return eid;
}
