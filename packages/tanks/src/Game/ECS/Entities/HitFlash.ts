import { addEntity } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { HitFlash } from '../Components/HitFlash.ts';
import { Progress } from '../Components/Progress.ts';
import { DestroyByTimeout } from '../Components/Destroy.ts';
import { addTransformComponents, applyMatrixTranslate, applyMatrixScale, LocalTransform } from '../../../../../renderer/src/ECS/Components/Transform.ts';
import { ZIndex } from '../../consts.ts';
import { RenderDI } from '../../DI/RenderDI.ts';
import { Sound, SoundType } from '../Components/Sound.ts';

export interface HitFlashOptions {
    x: number;
    y: number;
    size: number;
    duration: number;
}

export function spawnHitFlash(options: HitFlashOptions, { world } = GameDI, { enabled } = RenderDI) {
    if (!enabled) return;

    const eid = addEntity(world);

    addTransformComponents(world, eid);
    applyMatrixTranslate(LocalTransform.matrix.getBatch(eid), options.x, options.y, ZIndex.HitFlash);
    applyMatrixScale(LocalTransform.matrix.getBatch(eid), options.size, options.size);

    HitFlash.addComponent(world, eid);
    Progress.addComponent(world, eid, options.duration);
    DestroyByTimeout.addComponent(world, eid, options.duration);

    // Add hit sound - entity already has Transform for position
    Sound.addComponent(world, eid, SoundType.TankHit, {
        loop: false,
        volume: 1,
        autoplay: true,
    });
}
