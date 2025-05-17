import { PLAYER_REFS } from '../../consts.ts';
import { TankController } from '../Components/TankController.ts';
import { isNil } from 'lodash-es';
import { getMatrixTranslation, LocalTransform } from '../../../../../../src/ECS/Components/Transform.ts';
import { Tank } from '../Components/Tank.ts';
import { PlayerEnvDI } from '../../DI/PlayerEnvDI.ts';
import { RenderDI } from '../../DI/RenderDI.ts';

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
        if (PLAYER_REFS.tankPid) {
            TankController.setMove$(PLAYER_REFS.tankPid, move);
            TankController.setRotate$(PLAYER_REFS.tankPid, rotation);
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
        if (PLAYER_REFS.tankPid && lastEvent) {
            const currentPosition = getMatrixTranslation(LocalTransform.matrix.getBatch(Tank.aimEid[PLAYER_REFS.tankPid]));
            TankController.setTurretDir$(PLAYER_REFS.tankPid, lastEvent.clientX - currentPosition[0], lastEvent.clientY - currentPosition[1]);
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
        !isNil(PLAYER_REFS.tankPid) && TankController.setShooting$(PLAYER_REFS.tankPid, shooting);
    };

    const destroy = () => {
        document.removeEventListener('keydown', onKeyDown);
        document.removeEventListener('keyup', onKeyUp);
        canvas.removeEventListener('mousedown', onMouseDown);
        canvas.removeEventListener('mouseup', onMouseUp);
    };

    return { tick, destroy };
}
