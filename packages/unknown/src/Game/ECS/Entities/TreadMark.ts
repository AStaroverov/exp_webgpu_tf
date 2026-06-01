import { getFxWorldComponents } from '../createFxWorld.ts';
import { createRectangle } from '../../../../../renderer/src/ECS/Entities/Shapes.ts';
import {
    applyMatrixRotateZ,
} from '../../../../../renderer/src/ECS/Components/Transform.ts';
import { ZIndex } from '../../consts.ts';
import { RenderDI } from '../../DI/RenderDI.ts';
import { Worlds } from '../../DI/Worlds.ts';

export const TREAD_MARK_DURATION = 10_000;

const TREAD_MARK_COLOR: [number, number, number, number] = [0.35, 0.28, 0.2, 0.4];

export interface TreadMarkOptions {
    x: number;
    y: number;
    width: number;
    height: number;
    rotation?: number;
}

export function spawnTreadMark(options: TreadMarkOptions, { fxWorld } = Worlds, { enabled } = RenderDI) {
    if (!enabled) return;
    const { LocalTransform, TreadMark, ProgressFx, DestroyByTimeoutFx } = getFxWorldComponents(fxWorld);

    const eid = createRectangle(fxWorld, {
        x: options.x,
        y: options.y,
        z: ZIndex.TreadMark,
        width: options.width,
        height: options.height,
        color: TREAD_MARK_COLOR,
    });

    if (options.rotation) {
        applyMatrixRotateZ(LocalTransform.matrix.getBatch(eid), options.rotation);
    }

    TreadMark.addComponent(fxWorld, eid);
    ProgressFx.addComponent(fxWorld, eid, TREAD_MARK_DURATION);
    DestroyByTimeoutFx.addComponent(fxWorld, eid, TREAD_MARK_DURATION);
}
