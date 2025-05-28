import { TankController } from '../Components/TankController.ts';
import { clamp, isNil } from 'lodash-es';
import { Tank } from '../Components/Tank.ts';
import { PlayerEnvDI } from '../../DI/PlayerEnvDI.ts';
import { RenderDI } from '../../DI/RenderDI.ts';
import { RigidBodyState } from '../Components/Physical.ts';
import { normalizeAngle } from '../../../../../../lib/math.ts';

export function createPlayerTankPositionSystem({ document } = PlayerEnvDI) {
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
        if (PlayerEnvDI.tankEid) {
            TankController.setMove$(PlayerEnvDI.tankEid, move);
            TankController.setRotate$(PlayerEnvDI.tankEid, rotation);
        }
    };
    const destroy = () => {
        document.removeEventListener('keydown', onKeyDown);
        document.removeEventListener('keyup', onKeyUp);
    };

    return { tick, destroy };
}

export function createPlayerTankTurretRotationSystem({ document } = PlayerEnvDI) {
    let lastEvent: undefined | MouseEvent;
    let callback = (event: MouseEvent) => {
        lastEvent = event;
    };
    document.addEventListener('mousemove', callback);

    const tick = () => {
        if (PlayerEnvDI.tankEid && lastEvent) {
            const tankRot = RigidBodyState.rotation[PlayerEnvDI.tankEid];
            const turretRot = RigidBodyState.rotation[Tank.turretEId[PlayerEnvDI.tankEid]];
            const turretPos = RigidBodyState.position.getBatch(Tank.turretEId[PlayerEnvDI.tankEid]);

            // Глобальный угол от дула к позиции цели
            const targetRot = Math.atan2(lastEvent.clientY - turretPos[1], lastEvent.clientX - turretPos[0]) + Math.PI / 2;
            const relTurretRot = normalizeAngle(turretRot - tankRot);
            const relTargetTurretRot = normalizeAngle(targetRot - tankRot);
            const deltaRot = normalizeAngle(relTargetTurretRot - relTurretRot);

            TankController.setTurretRotation$(PlayerEnvDI.tankEid, clamp(deltaRot, -1, 1));
        }
    };
    const destroy = () => {
        document.removeEventListener('mousemove', callback);
        lastEvent = undefined;
    };

    return { tick, destroy };
}

export function createPlayerTankBulletSystem({ document } = PlayerEnvDI, { canvas } = RenderDI,
) {
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
    canvas.addEventListener('mousedown', onMouseDown);
    canvas.addEventListener('mouseup', onMouseUp);

    const tick = () => {
        !isNil(PlayerEnvDI.tankEid) && TankController.setShooting$(PlayerEnvDI.tankEid, shooting);
    };

    const destroy = () => {
        document.removeEventListener('keydown', onKeyDown);
        document.removeEventListener('keyup', onKeyUp);
        canvas.removeEventListener('mousedown', onMouseDown);
        canvas.removeEventListener('mouseup', onMouseUp);
    };

    return { tick, destroy };
}
