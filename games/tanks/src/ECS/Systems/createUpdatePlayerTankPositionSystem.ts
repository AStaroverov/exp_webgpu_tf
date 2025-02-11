// import { DI } from '../../DI';
// import { RigidBodyRef } from '../Components/Physical.ts';
// import { Vector2 } from '@dimforge/rapier2d';
// import { applyRotationToVector } from '../../Physical/applyRotationToVector.ts';
//
// export function createUpdatePlayerTankPositionSystem(tankId: number, { document, physicalWorld } = DI) {
//     let speed = 0;
//     let rotation = 0;
//
//     let dSpeed = 0;
//     let dRotation = 0;
//
//     const maxAbsSpeed = 1;
//     const maxAbsRotation = 0.1;
//
//     const nextLinvel = new Vector2(0, 0);
//
//     document.addEventListener('keydown', (event) => {
//         switch (event.key) {
//             case 'w':
//             case 'ArrowUp': {
//                 dSpeed = 0.1;
//                 break;
//             }
//             case 's':
//             case 'ArrowDown': {
//                 dSpeed = -0.1;
//                 break;
//             }
//             case 'a':
//             case 'ArrowLeft': {
//                 dRotation = -0.005;
//                 break;
//             }
//             case 'd':
//             case 'ArrowRight': {
//                 dRotation = 0.005;
//                 break;
//             }
//         }
//     });
//
//     document.addEventListener('keyup', (event) => {
//         switch (event.key) {
//             case 'w':
//             case 's':
//             case 'ArrowUp':
//             case 'ArrowDown': {
//                 dSpeed = 0;
//                 break;
//             }
//             case 'a':
//             case 'd':
//             case 'ArrowLeft':
//             case 'ArrowRight': {
//                 dRotation = 0;
//                 break;
//             }
//         }
//     });
//
//     return function () {
//         if (speed < maxAbsSpeed && speed > -maxAbsSpeed) {
//             speed += dSpeed;
//         }
//         if (rotation < maxAbsRotation && rotation > -maxAbsRotation) {
//             rotation += dRotation;
//         }
//
//         const rb = physicalWorld.getRigidBody(RigidBodyRef.id[tankId]);
//         const nextRotation = rb.rotation() + rotation;
//         // const nextRotation = rb.rotation();
//         // const nextTranslation = rb.translation();
//
//         nextLinvel.x = 0;
//         nextLinvel.y = -speed * 800;
//         applyRotationToVector(nextLinvel, nextLinvel, nextRotation);
//
//         // rb.setLinvel(nextLinvel, true);
//         // rb.setAngvel(rotation * 1000, true);
//
//         nextLinvel.x *= 10000;
//         nextLinvel.y *= 10000;
//         console.log('>>', rb.mass());
//
//         rb.applyImpulse(nextLinvel, true);
//         rb.applyTorqueImpulse(nextRotation * 100000, true);
//         //
//         // nextTranslation.y += speed * 80;
//         // rb.setNextKinematicRotation(nextRotation);
//         // rb.setNextKinematicTranslation(nextTranslation);
//
//         // speed *= 0.9;
//         // rotation *= 0.9;
//
//         speed = 0;
//         rotation = 0;
//     };
// }
import { DI } from '../../DI';
import { RigidBodyRef } from '../Components/Physical.ts';
import { Vector2 } from '@dimforge/rapier2d';
import { applyRotationToVector } from '../../Physical/applyRotationToVector.ts';

export function createUpdatePlayerTankPositionSystem(tankId: number, { document, physicalWorld } = DI) {
    let speed = 0;
    let rotation = 0;

    const acceleration = 0.1; // Как быстро набирается скорость
    const maxSpeed = 1.5; // Максимальная скорость танка
    const rotationSpeed = 0.003; // Скорость поворота
    const impulseFactor = 500000; // Масштаб импульса (настраиваемый)
    const rotationImpulseFactor = 1000000000; // Масштаб крутящего момента

    let moveDirection = 0;
    let rotationDirection = 0;

    const nextLinvel = new Vector2(0, 0);

    document.addEventListener('keydown', (event) => {
        switch (event.key) {
            case 'w':
            case 'ArrowUp': {
                moveDirection = 1;
                break;
            }
            case 's':
            case 'ArrowDown': {
                moveDirection = -1;
                break;
            }
            case 'a':
            case 'ArrowLeft': {
                rotationDirection = -1;
                break;
            }
            case 'd':
            case 'ArrowRight': {
                rotationDirection = 1;
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
                moveDirection = 0;
                break;
            }
            case 'a':
            case 'd':
            case 'ArrowLeft':
            case 'ArrowRight': {
                rotationDirection = 0;
                break;
            }
        }
    });

    return function () {
        const rb = physicalWorld.getRigidBody(RigidBodyRef.id[tankId]);

        // Управление скоростью
        if (moveDirection !== 0) {
            speed += moveDirection * acceleration;
            speed = Math.max(-maxSpeed, Math.min(maxSpeed, speed));
        } else {
            speed *= 0.5; // Плавное замедление
        }

        // Управление поворотом
        if (rotationDirection !== 0) {
            rotation += rotationDirection * rotationSpeed;
        } else {
            rotation *= 0.5; // Плавное уменьшение поворота
        }

        // Получаем текущий угол поворота
        const nextRotation = rb.rotation();

        // Задаем направление движения
        nextLinvel.x = 0;
        nextLinvel.y = -speed * impulseFactor;
        applyRotationToVector(nextLinvel, nextLinvel, nextRotation);

        // Применяем импульс для движения
        rb.applyImpulse(nextLinvel, true);

        // Применяем крутящий момент для поворота
        rb.applyTorqueImpulse(rotation * rotationImpulseFactor, true);
    };
}
