import { once } from 'lodash-es';
import { DI } from '../../DI';
import { Bullet, createBulletRR } from '../Components/Bullet.ts';
import {
    getMatrixRotationZ,
    getMatrixTranslationX,
    getMatrixTranslationY,
    GlobalTransform,
} from '../../../../../src/ECS/Components/Transform.ts';
import { Tank } from '../Components/Tank.ts';
import { mat4, vec3 } from 'gl-matrix';
import { defineQuery, removeEntity } from 'bitecs';

export function createSpawnerBulletsSystem(tankId: number, { world, document, canvas } = DI) {
    const initSpawnBullet = once(() => {
        document.addEventListener('keypress', (event) => {
            switch (event.code) {
                case 'Space': {
                    spawnBullet(tankId);
                    break;
                }
            }
        });
    });

    const query = defineQuery([Bullet, GlobalTransform]);

    return (() => {
        initSpawnBullet();

        const { width, height } = canvas;
        const bulletsIds = query(world);

        for (let i = 0; i < bulletsIds.length; i++) {
            const bulletId = bulletsIds[i];
            const bulletGlobalTransform = GlobalTransform.matrix[bulletId];

            const x = getMatrixTranslationX(bulletGlobalTransform);
            const y = getMatrixTranslationY(bulletGlobalTransform);

            if (x < 0 || x > width || y < 0 || y > height) {
                removeEntity(world, bulletId);
            }
        }
    });
}

const mutatedOptions = {
    x: 0,
    y: 0,
    rotation: 0,
    speed: 0,
    color: [1, 0, 0, 1] as [number, number, number, number],
};

const tmpMatrix = mat4.create();
const tmpPosition = vec3.create() as Float32Array;

function spawnBullet(parentId: number) {
    const tankGlobalTransform = GlobalTransform.matrix[parentId];
    const bulletDelta = Tank.bulletStartPosition[parentId];

    tmpPosition.set(bulletDelta);
    mat4.identity(tmpMatrix);
    mat4.translate(tmpMatrix, tmpMatrix, tmpPosition);
    mat4.multiply(tmpMatrix, tankGlobalTransform, tmpMatrix);

    mutatedOptions.x = getMatrixTranslationX(tmpMatrix);
    mutatedOptions.y = getMatrixTranslationY(tmpMatrix);
    mutatedOptions.rotation = getMatrixRotationZ(tmpMatrix);
    mutatedOptions.speed = Tank.bulletSpeed[parentId];

    createBulletRR(mutatedOptions);
}
