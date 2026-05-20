import { EntityId } from 'bitecs';
import { GameDI } from '../../../DI/GameDI.ts';
import { GameMap } from '../../Entities/GameMap.ts';
import { getGameComponents } from '../../createGameWorld.ts';

export interface CameraState {
    x: number;
    y: number;
    targetEid: EntityId | null;
    smoothing: number;
    updateMapOffset: boolean;
}

export const CameraState: CameraState = {
    x: 0,
    y: 0,
    targetEid: null,
    smoothing: 0.08,
    updateMapOffset: false,
};

export function initCameraPosition() {
    CameraState.x = GameMap.offsetX;
    CameraState.y = GameMap.offsetY;
}

export function setCameraTarget(eid: EntityId | null, { world } = GameDI) {
    CameraState.targetEid = eid;

    if (eid !== null) {
        const { RigidBodyState } = getGameComponents(world);
        CameraState.x = RigidBodyState.position.get(eid, 0);
        CameraState.y = RigidBodyState.position.get(eid, 1);

        if (CameraState.updateMapOffset) {
            GameMap.setOffset(CameraState.x, CameraState.y);
        }
    }
}

export function setCameraSmoothing(smoothing: number) {
    CameraState.smoothing = Math.max(0.01, Math.min(1, smoothing));
}

export function setInfiniteMapMode(enabled: boolean) {
    CameraState.updateMapOffset = enabled;
}

export function getCameraPosition(): { x: number; y: number } {
    return { x: CameraState.x, y: CameraState.y };
}

export function createCameraSystem({ world } = GameDI) {
    const { RigidBodyState } = getGameComponents(world);

    return function updateCamera(_delta: number) {
        const { targetEid, smoothing, updateMapOffset } = CameraState;

        if (targetEid === null) {
            return;
        }

        const targetX = RigidBodyState.position.get(targetEid, 0);
        const targetY = RigidBodyState.position.get(targetEid, 1);

        CameraState.x += (targetX - CameraState.x) * smoothing;
        CameraState.y += (targetY - CameraState.y) * smoothing;

        if (updateMapOffset) {
            GameMap.setOffset(CameraState.x, CameraState.y);
        }
    };
}
