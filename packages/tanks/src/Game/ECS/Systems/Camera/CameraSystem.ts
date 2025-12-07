import { EntityId } from 'bitecs';
import { RigidBodyState } from '../../Components/Physical.ts';
import { GameMap } from '../../Entities/GameMap.ts';

export interface CameraState {
    // Current camera position (what we render)
    x: number;
    y: number;
    // Target entity to follow
    targetEid: EntityId | null;
    // Smoothing factor (0-1, lower = smoother)
    smoothing: number;
    // Whether to update GameMap offset (for infinite map mode)
    updateMapOffset: boolean;
}

export const CameraState: CameraState = {
    x: 0,
    y: 0,
    targetEid: null,
    smoothing: 0.08,
    updateMapOffset: false,
};

/**
 * Initialize camera position from GameMap offset.
 * Should be called after GameMap.setOffset in createGame.
 */
export function initCameraPosition() {
    CameraState.x = GameMap.offsetX;
    CameraState.y = GameMap.offsetY;
}

export function setCameraTarget(eid: EntityId | null) {
    CameraState.targetEid = eid;
    
    // If setting a new target, snap camera to it immediately
    if (eid !== null) {
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

export function createCameraSystem() {
    return function updateCamera(_delta: number) {
        const { targetEid, smoothing, updateMapOffset } = CameraState;
        
        if (targetEid === null) {
            return;
        }
        
        // Get target position
        const targetX = RigidBodyState.position.get(targetEid, 0);
        const targetY = RigidBodyState.position.get(targetEid, 1);
        
        // Lerp camera position towards target
        CameraState.x += (targetX - CameraState.x) * smoothing;
        CameraState.y += (targetY - CameraState.y) * smoothing;
        
        // Update GameMap offset if in infinite map mode
        if (updateMapOffset) {
            GameMap.setOffset(CameraState.x, CameraState.y);
        }
    };
}
