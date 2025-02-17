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
import { Player } from '../Components/Player.ts';

export function createSpawnerBulletsSystem(tankId: number, { world, document, canvas } = DI) {
    const initSpawnBullet = once(() => {
        document.addEventListener('keypress', (event) => {
            event.preventDefault();
            switch (event.code) {
                case 'Space': {
                    spawnBullet(tankId);
                    break;
                }
            }
        });

        canvas.addEventListener('click', (event) => {
            event.preventDefault();
            spawnBullet(tankId);
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
    width: 5,
    height: 5,
    rotation: 0,
    speed: 0,
    mass: 100,
    color: new Float32Array([1, 0, 0, 1]),
    playerId: 0,
    shadow: new Float32Array([0, 8]),
};

const tmpMatrix = mat4.create();
const tmpPosition = vec3.create() as Float32Array;

function spawnBullet(tankId: number) {
    const globalTransform = GlobalTransform.matrix[Tank.turretEId[tankId]];
    const bulletDelta = Tank.bulletStartPosition[tankId];

    tmpPosition.set(bulletDelta);
    mat4.identity(tmpMatrix);
    mat4.translate(tmpMatrix, tmpMatrix, tmpPosition);
    mat4.multiply(tmpMatrix, globalTransform, tmpMatrix);

    mutatedOptions.x = getMatrixTranslationX(tmpMatrix);
    mutatedOptions.y = getMatrixTranslationY(tmpMatrix);
    mutatedOptions.rotation = getMatrixRotationZ(tmpMatrix);
    mutatedOptions.speed = Tank.bulletSpeed[tankId];
    mutatedOptions.playerId = Player.id[tankId];

    createBulletRR(mutatedOptions);
}
