import { DI } from '../../DI';
import { PLAYER_REFS } from '../../consts.ts';
import { TankController } from '../Components/TankController.ts';
import { isNil } from 'lodash-es';
import { getMatrixTranslation, LocalTransform } from '../../../../../src/ECS/Components/Transform.ts';
import { Tank } from '../Components/Tank.ts';

export function createPlayerTankPositionSystem({ document } = DI) {
    document.addEventListener('keydown', (event) => {
        if (isNil(PLAYER_REFS.tankPid)) return;
        switch (event.key) {
            case 'w':
            case 'ArrowUp': {
                TankController.setMove$(PLAYER_REFS.tankPid, 1);
                event.preventDefault();
                break;
            }
            case 's':
            case 'ArrowDown': {
                TankController.setMove$(PLAYER_REFS.tankPid, -1);
                event.preventDefault();
                break;
            }
            case 'a':
            case 'ArrowLeft': {
                TankController.setRotate$(PLAYER_REFS.tankPid, -1);
                event.preventDefault();
                break;
            }
            case 'd':
            case 'ArrowRight': {
                TankController.setRotate$(PLAYER_REFS.tankPid, 1);
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
                TankController.setMove$(tankPid, 0);
                break;
            }
            case 'a':
            case 'd':
            case 'ArrowLeft':
            case 'ArrowRight': {
                TankController.setRotate$(tankPid, 0);
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
        const currentPosition = getMatrixTranslation(LocalTransform.matrix.getBatche(Tank.aimEid[PLAYER_REFS.tankPid]));
        TankController.setTurretDir$(PLAYER_REFS.tankPid, event.clientX - currentPosition[0], event.clientY - currentPosition[1]);
    });

    return () => {
    };
}

export function createPlayerTankBulletSystem({ document, canvas } = DI) {
    document.addEventListener('keydown', (event) => {
        event.preventDefault();
        switch (event.code) {
            case 'Space': {
                !isNil(PLAYER_REFS.tankPid) && TankController.setShooting$(PLAYER_REFS.tankPid, true);
                break;
            }
        }
    });
    document.addEventListener('keyup', (event) => {
        event.preventDefault();
        switch (event.code) {
            case 'Space': {
                !isNil(PLAYER_REFS.tankPid) && TankController.setShooting$(PLAYER_REFS.tankPid, false);
                break;
            }
        }
    });

    canvas.addEventListener('mousedown', (event) => {
        event.preventDefault();
        !isNil(PLAYER_REFS.tankPid) && TankController.setShooting$(PLAYER_REFS.tankPid, true);
    });

    canvas.addEventListener('mouseup', (event) => {
        event.preventDefault();
        !isNil(PLAYER_REFS.tankPid) && TankController.setShooting$(PLAYER_REFS.tankPid, false);
    });

    return () => {
    };
}
