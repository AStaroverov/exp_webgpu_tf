import { addEntity, EntityId } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { HitFlash } from '../Components/HitFlash.ts';
import { DestroyByTimeout } from '../Components/Destroy.ts';

export interface HitFlashOptions {
    x: number;
    y: number;
    size: number;
    duration: number;
}

export function createHitFlash(options: HitFlashOptions, { world } = GameDI): EntityId {
    const eid = addEntity(world);

    HitFlash.addComponent(
        world,
        eid,
        options.x,
        options.y,
        options.size,
        options.duration,
    );
    DestroyByTimeout.addComponent(world, eid, options.duration);

    return eid;
}
