import { addEntity, EntityId } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { Explosion } from '../Components/Explosion.ts';
import { DestroyByTimeout } from '../Components/Destroy.ts';

export interface ExplosionOptions {
    x: number;
    y: number;
    size: number;
    duration: number;
}

export function createExplosion(options: ExplosionOptions, { world } = GameDI): EntityId {
    const eid = addEntity(world);

    Explosion.addComponent(
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
