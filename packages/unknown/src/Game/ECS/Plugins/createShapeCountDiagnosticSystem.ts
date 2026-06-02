/**
 * TEMPORARY diagnostic — counts renderable SDF-shape entities and the main
 * dynamic contributors once per second, to find what overflows the 10k instance
 * buffer (MAX_INSTANCE_COUNT in sdf.shader.ts). DELETE this file + its wiring in
 * createGame.ts once the leak is identified.
 */

import { query } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { getGameComponents } from '../createGameWorld.ts';

export function createShapeCountDiagnosticSystem({ world } = GameDI) {
    const { Shape, Color, GlobalTransform, Bullet, TreadMark, DestroyByTimeout } =
        getGameComponents(world);
    let acc = 0;

    return function diagnostic(delta: number) {
        acc += delta;
        if (acc < 1000) return;
        acc = 0;

        const sdf = query(world, [GlobalTransform, Shape, Color]).length;
        const bullets = query(world, [Bullet]).length;
        const tread = query(world, [TreadMark]).length;
        const timed = query(world, [DestroyByTimeout]).length;
        const allShapes = query(world, [Shape]).length;

        // eslint-disable-next-line no-console
        console.warn(
            `[shape-count] SDF(GT+Shape+Color)=${sdf} | allShape=${allShapes} | ` +
                `bullets=${bullets} treadMarks=${tread} destroyByTimeout=${timed} | cap=10000`,
        );
    };
}
