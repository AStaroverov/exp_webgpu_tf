import { TankController } from '../Components/TankController.ts';
import { clamp } from 'lodash-es';
import { Tank } from '../Components/Tank.ts';
import { PlayerEnvDI } from '../../DI/PlayerEnvDI.ts';
import { RenderDI } from '../../DI/RenderDI.ts';
import { RigidBodyState } from '../Components/Physical.ts';
import { normalizeAngle } from '../../../../../../lib/math.ts';
import { hasComponent } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';

function getPlayerTankEid({ world } = GameDI) {
    return PlayerEnvDI.tankEid && hasComponent(world, PlayerEnvDI.tankEid, TankController)
        ? PlayerEnvDI.tankEid
        : null;
}

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
        const tankEid = getPlayerTankEid();
        if (tankEid) {
            TankController.setMove$(tankEid, move);
            TankController.setRotate$(tankEid, rotation);
        }
    };
    const destroy = () => {
        document.removeEventListener('keydown', onKeyDown);
        document.removeEventListener('keyup', onKeyUp);
    };

    return { tick, destroy };
}

export function createPlayerTankTurretRotationSystem({ canvas } = RenderDI) {
    let lastEvent: undefined | MouseEvent;
    let callback = (event: MouseEvent) => {
        lastEvent = event;
    };
    canvas.addEventListener('mousemove', callback);

    const tick = () => {
        const tankEid = getPlayerTankEid();
        if (tankEid && lastEvent) {
            const tankRot = RigidBodyState.rotation[tankEid];
            const turretRot = RigidBodyState.rotation[Tank.turretEId[tankEid]];
            const turretPos = RigidBodyState.position.getBatch(Tank.turretEId[tankEid]);

            const targetRot = Math.atan2(lastEvent.offsetY - turretPos[1], lastEvent.offsetX - turretPos[0]) + Math.PI / 2;
            const relTurretRot = normalizeAngle(turretRot - tankRot);
            const relTargetTurretRot = normalizeAngle(targetRot - tankRot);
            const deltaRot = normalizeAngle(relTargetTurretRot - relTurretRot);

            TankController.setTurretRotation$(tankEid, clamp(deltaRot, -1, 1));
        }
    };
    const destroy = () => {
        canvas.removeEventListener('mousemove', callback);
        lastEvent = undefined;
    };

    return { tick, destroy };
}

export function createPlayerTankBulletSystem({ document } = PlayerEnvDI, { canvas } = RenderDI) {
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
        const tankEid = getPlayerTankEid();
        tankEid && TankController.setShooting$(tankEid, shooting);
    };

    const destroy = () => {
        document.removeEventListener('keydown', onKeyDown);
        document.removeEventListener('keyup', onKeyUp);
        canvas.removeEventListener('mousedown', onMouseDown);
        canvas.removeEventListener('mouseup', onMouseUp);
    };

    return { tick, destroy };
}
