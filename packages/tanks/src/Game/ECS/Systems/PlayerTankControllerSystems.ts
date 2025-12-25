import { Tank } from '../Components/Tank.ts';
import { TurretController } from '../Components/TurretController.ts';
import { VehicleController } from '../Components/VehicleController.ts';
import { clamp, isNil } from 'lodash-es';
import { PlayerEnvDI } from '../../DI/PlayerEnvDI.ts';
import { RenderDI } from '../../DI/RenderDI.ts';
import { RigidBodyState } from '../Components/Physical.ts';
import { normalizeAngle } from '../../../../../../lib/math.ts';
import { GameMap } from '../Entities/GameMap.ts';
import { entityExists, hasComponent } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';

export function createPlayerTankPositionSystem({ world } = GameDI) {
    let move = 0;
    let rotation = 0;
    const onKeyDown = (event: KeyboardEvent) => {
        switch (event.key) {
            case 'w':
            case 'ArrowUp': {
                move = 1;
                event.preventDefault();
                break;
            }
            case 's':
            case 'ArrowDown': {
                move = -1;
                event.preventDefault();
                break;
            }
            case 'a':
            case 'ArrowLeft': {
                rotation = -1;
                event.preventDefault();
                break;
            }
            case 'd':
            case 'ArrowRight': {
                rotation = 1;
                event.preventDefault();
                break;
            }
        }
    };
    const onKeyUp = (event: KeyboardEvent) => {
        switch (event.key) {
            case 'w':
            case 's':
            case 'ArrowUp':
            case 'ArrowDown': {
                move = 0;
                break;
            }
            case 'a':
            case 'd':
            case 'ArrowLeft':
            case 'ArrowRight': {
                rotation = 0;
                break;
            }
        }
    };

    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);

    const tick = () => {
        if (isPlayerExist()) {
            const tankEid = PlayerEnvDI.tankEid!;
            
            if (hasComponent(world, tankEid, VehicleController)) {
                VehicleController.setMove$(tankEid, move);
                VehicleController.setRotate$(tankEid, rotation);
            }
        }
    };
    const destroy = () => {
        document.removeEventListener('keydown', onKeyDown);
        document.removeEventListener('keyup', onKeyUp);
    };

    return { tick, destroy };
}

export function createPlayerTankTurretRotationSystem({ world } = GameDI) {
    let lastEvent: undefined | MouseEvent;
    let callback = (event: MouseEvent) => {
        lastEvent = event;
    };
    document.addEventListener('mousemove', callback);

    const tick = () => {
        if (isPlayerExist() && entityExists(world, Tank.turretEId[PlayerEnvDI.tankEid!]) && lastEvent) {
            const vehicleRot = RigidBodyState.rotation[PlayerEnvDI.tankEid!];
            const turretRot = RigidBodyState.rotation[Tank.turretEId[PlayerEnvDI.tankEid!]];
            const turretPos = RigidBodyState.position.getBatch(Tank.turretEId[PlayerEnvDI.tankEid!]);

            // Convert screen coordinates to world coordinates
            const [worldX, worldY] = screenToWorld(lastEvent.clientX, lastEvent.clientY);

            // Глобальный угол от дула к позиции цели
            const targetRot = Math.atan2(worldY - turretPos[1], worldX - turretPos[0]);
            const relTurretRot = normalizeAngle(turretRot - vehicleRot);
            const relTargetTurretRot = normalizeAngle(targetRot - vehicleRot);
            const deltaRot = normalizeAngle(relTargetTurretRot - relTurretRot);

            const turretEid = Tank.turretEId[PlayerEnvDI.tankEid!];
            TurretController.setRotation$(turretEid, clamp(deltaRot, -1, 1));}
    };
    const destroy = () => {
        document.removeEventListener('mousemove', callback);
        lastEvent = undefined;
    };

    return { tick, destroy };
}

export function createPlayerTankBulletSystem({ world } = GameDI) {
    let shooting = 0;
    const onKeyDown = (event: KeyboardEvent) => {
        event.preventDefault();
        switch (event.code) {
            case 'Space': {
                shooting = 1;
                break;
            }
        }
    };
    const onKeyUp = (event: KeyboardEvent) => {
        event.preventDefault();
        switch (event.code) {
            case 'Space': {
                shooting = -1;
                break;
            }
        }
    };
    const onMouseDown = (event: MouseEvent) => {
        event.preventDefault();
        shooting = 1;
    };
    const onMouseUp = (event: MouseEvent) => {
        event.preventDefault();
        shooting = -1;
    };

    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);
    document.addEventListener('mousedown', onMouseDown);
    document.addEventListener('mouseup', onMouseUp);

    const tick = () => {
        if (isPlayerExist() && entityExists(world, Tank.turretEId[PlayerEnvDI.tankEid!])) {
            const turretEid = Tank.turretEId[PlayerEnvDI.tankEid!];
            TurretController.setShooting$(turretEid, shooting);
        }
    };

    const destroy = () => {
        document.removeEventListener('keydown', onKeyDown);
        document.removeEventListener('keyup', onKeyUp);
        document.removeEventListener('mousedown', onMouseDown);
        document.removeEventListener('mouseup', onMouseUp);
    };

    return { tick, destroy };
}

function isPlayerExist({ world } = GameDI, { tankEid } = PlayerEnvDI) {
    return !isNil(tankEid) && entityExists(world, tankEid)
}

function screenToWorld(screenX: number, screenY: number): [number, number] {
    if (!RenderDI.canvas) return [screenX, screenY];

    const rect = RenderDI.canvas.getBoundingClientRect();
    
    // Convert screen position to normalized position relative to canvas center (-0.5 to 0.5)
    const normalizedX = (screenX - rect.left) / rect.width - 0.5;
    const normalizedY = (screenY - rect.top) / rect.height - 0.5;
    
    // Convert to world coordinates relative to camera (canvas center = camera position)
    const worldX = GameMap.offsetX + normalizedX * rect.width;
    const worldY = GameMap.offsetY + normalizedY * rect.height;

    return [worldX, worldY];
}
