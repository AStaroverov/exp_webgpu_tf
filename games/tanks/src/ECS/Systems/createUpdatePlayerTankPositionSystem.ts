import { DI } from '../../DI';
import {
    applyMatrixRotateZ,
    applyMatrixTranslate,
    LocalTransform,
} from '../../../../../src/ECS/Components/Transform.ts';

export function createUpdatePlayerTankPositionSystem(tankId: number, { document } = DI) {
    let speed = 0;
    let rotation = 0;

    let dSpeed = 0;
    let dRotation = 0;

    const maxAbsSpeed = 1;
    const maxAbsRotation = 0.1;

    document.addEventListener('keydown', (event) => {
        switch (event.key) {
            case 'w':
            case 'ArrowUp': {
                dSpeed = 0.1;
                break;
            }
            case 's':
            case 'ArrowDown': {
                dSpeed = -0.1;
                break;
            }
            case 'a':
            case 'ArrowLeft': {
                dRotation = -0.005;
                break;
            }
            case 'd':
            case 'ArrowRight': {
                dRotation = 0.005;
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
                dSpeed = 0;
                break;
            }
            case 'a':
            case 'd':
            case 'ArrowLeft':
            case 'ArrowRight': {
                dRotation = 0;
                break;
            }
        }
    });

    return function () {
        if (speed < maxAbsSpeed && speed > -maxAbsSpeed) {
            speed += dSpeed;
        }
        if (rotation < maxAbsRotation && rotation > -maxAbsRotation) {
            rotation += dRotation;
        }

        applyMatrixTranslate(LocalTransform.matrix[tankId], 0, -speed);
        applyMatrixRotateZ(LocalTransform.matrix[tankId], rotation);

        speed *= 0.9;
        rotation *= 0.9;
    };
}