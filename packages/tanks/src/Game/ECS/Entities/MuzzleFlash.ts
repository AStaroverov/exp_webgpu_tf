import { addEntity, EntityId } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { MuzzleFlash } from '../Components/MuzzleFlash.ts';
import { DestroyByTimeout } from '../Components/Destroy.ts';

export interface MuzzleFlashOptions {
    x: number;
    y: number;
    size: number;
    duration: number;
    rotation?: number;
}

export function createMuzzleFlash(options: MuzzleFlashOptions, { world } = GameDI): EntityId {
    const eid = addEntity(world);

    MuzzleFlash.addComponent(
        world,
        eid,
        options.x,
        options.y,
        options.size,
        options.duration,
        options.rotation ?? 0,
    );
    DestroyByTimeout.addComponent(world, eid, options.duration);

    return eid;
}
