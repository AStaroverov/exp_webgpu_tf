import { PLAYER_REFS } from '../../consts.ts';
import { TankController } from '../Components/TankController.ts';
import { isNil } from 'lodash-es';
import { getMatrixTranslation, LocalTransform } from '../../../../../src/ECS/Components/Transform.ts';
import { Tank } from '../Components/Tank.ts';
import { PlayerEnvDI } from '../../DI/PlayerEnvDI.ts';

export function createPlayerTankPositionSystem({ document } = PlayerEnvDI) {
    let move = 0;
    let rotation = 0;

    document.addEventListener('keydown', (event) => {
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
    });

    document.addEventListener('keyup', (event) => {
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
    });

    return () => {
        if (PLAYER_REFS.tankPid) {
            TankController.setMove$(PLAYER_REFS.tankPid, move);
            TankController.setRotate$(PLAYER_REFS.tankPid, rotation);
        }
    };
}

export function createPlayerTankTurretRotationSystem({ document } = PlayerEnvDI) {
    let lastEvent: undefined | MouseEvent;
    document.addEventListener('mousemove', (event) => {
        lastEvent = event;
    });

    return () => {
        if (PLAYER_REFS.tankPid && lastEvent) {
            const currentPosition = getMatrixTranslation(LocalTransform.matrix.getBatch(Tank.aimEid[PLAYER_REFS.tankPid]));
            TankController.setTurretDir$(PLAYER_REFS.tankPid, lastEvent.clientX - currentPosition[0], lastEvent.clientY - currentPosition[1]);
        }
    };
}

export function createPlayerTankBulletSystem({ document, container } = PlayerEnvDI) {
    let shooting = false;
    document.addEventListener('keydown', (event) => {
        event.preventDefault();
        switch (event.code) {
            case 'Space': {
                shooting = true;
                break;
            }
        }
    });
    document.addEventListener('keyup', (event) => {
        event.preventDefault();
        switch (event.code) {
            case 'Space': {
                shooting = false;
                break;
            }
        }
    });

    container.addEventListener('mousedown', (event) => {
        event.preventDefault();
        shooting = true;
    });

    container.addEventListener('mouseup', (event) => {
        event.preventDefault();
        shooting = false;
    });

    return () => {
        !isNil(PLAYER_REFS.tankPid) && TankController.setShooting$(PLAYER_REFS.tankPid, shooting);
    };
}
