import { DI } from '../../DI';
import { PLAYER_REFS } from '../../consts.ts';
import {
    setTankControllerMove,
    setTankControllerRotate,
    setTankControllerShot,
    setTankControllerTurretTarget,
} from '../Components/TankController.ts';
import { isNil } from 'lodash-es';

export function createPlayerTankPositionSystem({ document } = DI) {
    document.addEventListener('keydown', (event) => {
        if (isNil(PLAYER_REFS.tankPid)) return;
        switch (event.key) {
            case 'w':
            case 'ArrowUp': {
                setTankControllerMove(PLAYER_REFS.tankPid, 1);
                event.preventDefault();
                break;
            }
            case 's':
            case 'ArrowDown': {
                setTankControllerMove(PLAYER_REFS.tankPid, -1);
                event.preventDefault();
                break;
            }
            case 'a':
            case 'ArrowLeft': {
                setTankControllerRotate(PLAYER_REFS.tankPid, -1);
                event.preventDefault();
                break;
            }
            case 'd':
            case 'ArrowRight': {
                setTankControllerRotate(PLAYER_REFS.tankPid, 1);
                event.preventDefault();
                break;
            }
        }
    });

    document.addEventListener('keyup', (event) => {
        const { tankPid } = PLAYER_REFS;
        if (tankPid == null) return;

        switch (event.key) {
            case 'w':
            case 's':
            case 'ArrowUp':
            case 'ArrowDown': {
                setTankControllerMove(tankPid, 0);
                break;
            }
            case 'a':
            case 'd':
            case 'ArrowLeft':
            case 'ArrowRight': {
                setTankControllerRotate(tankPid, 0);
                break;
            }
        }
    });

    return () => {

    };
}

export function createPlayerTankTurretRotationSystem({ document } = DI) {
    document.addEventListener('mousemove', (event) => {
        if (isNil(PLAYER_REFS.tankPid)) return;
        setTankControllerTurretTarget(PLAYER_REFS.tankPid, event.clientX, event.clientY);
    });

    return () => {
    };
}

export function createPlayerTankBulletSystem({ document, canvas } = DI) {
    document.addEventListener('keypress', (event) => {
        event.preventDefault();
        switch (event.code) {
            case 'Space': {
                !isNil(PLAYER_REFS.tankPid) && setTankControllerShot(PLAYER_REFS.tankPid);
                break;
            }
        }
    });

    canvas.addEventListener('click', (event) => {
        event.preventDefault();
        !isNil(PLAYER_REFS.tankPid) && setTankControllerShot(PLAYER_REFS.tankPid);
    });

    return () => {
    };
}
