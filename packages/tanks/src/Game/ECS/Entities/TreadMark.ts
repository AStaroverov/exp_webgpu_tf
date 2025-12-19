import { GameDI } from '../../DI/GameDI.ts';
import { TreadMark } from '../Components/TreadMark.ts';
import { Progress } from '../Components/Progress.ts';
import { DestroyByTimeout } from '../Components/Destroy.ts';
import { createRectangle } from '../../../../../renderer/src/ECS/Entities/Shapes.ts';
import {
    LocalTransform,
    applyMatrixRotateZ,
} from '../../../../../renderer/src/ECS/Components/Transform.ts';
import { ZIndex } from '../../consts.ts';
import { RenderDI } from '../../DI/RenderDI.ts';

export const TREAD_MARK_DURATION = 10_000; // 10 seconds

// Dark brown tread mark color
const TREAD_MARK_COLOR: [number, number, number, number] = [0.35, 0.28, 0.2, 0.4];

export interface TreadMarkOptions {
    x: number;
    y: number;
    width: number;
    height: number;
    rotation?: number;
}

export function spawnTreadMark(options: TreadMarkOptions, { world } = GameDI, { enabled } = RenderDI) {
    if (!enabled) return;
    // Use the standard shape rendering system
    const eid = createRectangle(world, {
        x: options.x,
        y: options.y,
        z: ZIndex.TreadMark,
        width: options.width,
        height: options.height,
        color: TREAD_MARK_COLOR,
    });

    // Apply rotation if provided
    if (options.rotation) {
        applyMatrixRotateZ(LocalTransform.matrix.getBatch(eid), options.rotation);
    }

    // Add TreadMark and Progress components
    TreadMark.addComponent(world, eid);
    Progress.addComponent(world, eid, TREAD_MARK_DURATION);
    DestroyByTimeout.addComponent(world, eid, TREAD_MARK_DURATION);
}

