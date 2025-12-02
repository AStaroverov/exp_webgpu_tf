import { EntityId } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { TankTrack } from '../Components/TankTrack.ts';
import { DestroyByTimeout } from '../Components/Destroy.ts';
import { createRectangle } from '../../../../../renderer/src/ECS/Entities/Shapes.ts';
import {
    LocalTransform,
    applyMatrixRotateZ,
} from '../../../../../renderer/src/ECS/Components/Transform.ts';
import { ZIndex } from '../../consts.ts';

export const TRACK_DURATION = 10_000; // 10 seconds

// Dark brown track color
const TRACK_COLOR: [number, number, number, number] = [0.35, 0.28, 0.2, 0.4];

export interface TankTrackOptions {
    x: number;
    y: number;
    width: number;
    height: number;
    rotation?: number;
}

export function createTankTrack(options: TankTrackOptions, { world } = GameDI): EntityId {
    // Use the standard shape rendering system
    const eid = createRectangle(world, {
        x: options.x,
        y: options.y,
        z: ZIndex.TankTrack,
        width: options.width,
        height: options.height,
        color: TRACK_COLOR,
    });

    // Apply rotation if provided
    if (options.rotation) {
        applyMatrixRotateZ(LocalTransform.matrix.getBatch(eid), options.rotation);
    }

    // Add TankTrack component for age tracking
    TankTrack.addComponent(world, eid, TRACK_DURATION);
    DestroyByTimeout.addComponent(world, eid, TRACK_DURATION);

    return eid;
}
