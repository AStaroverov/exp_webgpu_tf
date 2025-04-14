import { ZIndex } from '../../../consts.ts';
import { CollisionGroup } from '../../../Physical/createRigid.ts';
import { TColor } from '../../../../../../src/ECS/Components/Common.ts';
import { JointData, Vector2 } from '@dimforge/rapier2d-simd';
import { GameDI } from '../../../DI/GameDI.ts';
import { RigidBodyRef } from '../Physical.ts';
import { createRectangleRR } from '../RigidRender.ts';
import { Player } from '../Player.ts';
import { Hitable } from '../Hitable.ts';
import { Parent } from '../Parent.ts';
import { Children } from '../Children.ts';
import { TankPart } from '../TankPart.ts';
import { BASE_DENSITY, Options, updateColorOptions } from './Options.ts';

export const SIZE = 5;
export const PADDING = SIZE + 1;
const hullSet: [number, number, number, number][] =
    Array.from({ length: 88 }, (_, i) => {
        return [
            i * PADDING % (PADDING * 8), Math.floor(i / 8) * PADDING,
            SIZE, SIZE,
        ];
    });
const turretSet: [number, number, number, number][] =
    Array.from({ length: 42 }, (_, i): [number, number, number, number] => {
        return [
            -PADDING * 2 + i * PADDING % (PADDING * 6), 10 * PADDING + Math.floor(i / 6) * PADDING,
            SIZE, SIZE,
        ];
    });
const gunSet: [number, number, number, number][] =
    Array.from({ length: 20 }, (_, i): [number, number, number, number] => {
        return [
            i * PADDING % (PADDING * 2), Math.floor(i / 2) * PADDING,
            SIZE, SIZE,
        ];
    });
const caterpillarSet: [number, number, number, number][] =
    Array.from({ length: 26 }, (_, i) => {
        return [
            i * PADDING % (PADDING * 2), Math.floor(i / 2) * PADDING,
            SIZE, SIZE,
        ];
    });

export const PARTS_COUNT = hullSet.length + turretSet.length + gunSet.length + caterpillarSet.length * 2;

/**
 * Создает корпус танка
 */
export function createTankHullParts(options: Options, tankEid: number, color: TColor): void {
    // Настройка опций для корпуса
    options.z = ZIndex.TankHull;
    options.density = BASE_DENSITY * 10;
    options.belongsSolverGroup = CollisionGroup.ALL;
    options.interactsSolverGroup = CollisionGroup.ALL;
    options.belongsCollisionGroup = CollisionGroup.TANK_HULL_PARTS;
    options.interactsCollisionGroup = CollisionGroup.BULLET | CollisionGroup.WALL | CollisionGroup.TANK_HULL_PARTS;

    // Создание корпуса с указанным цветом
    updateColorOptions(options, color);
    createRectanglesRR(tankEid, hullSet, options, 0 - 3.5 * PADDING, 0 - 5 * PADDING);
}

/**
 * Создает гусеницы танка (левую и правую)
 */
export function createTankTracksParts(options: Options, tankEid: number): void {
    // Настройка цвета для гусениц
    updateColorOptions(options, [0.5, 0.5, 0.5, 1]);

    // Создание левой гусеницы
    createRectanglesRR(tankEid, caterpillarSet, options, 0 - 5.5 * PADDING, 0 - 6 * PADDING);

    // Создание правой гусеницы
    createRectanglesRR(tankEid, caterpillarSet, options, 0 + 4.5 * PADDING, 0 - 6 * PADDING);
}

/**
 * Создает турель и орудие танка
 */
export function createTankTurretAndGunParts(options: Options, turretEid: number) {
    // Настройка опций для турели
    options.z = ZIndex.TankTurret;
    options.density = BASE_DENSITY;
    options.belongsCollisionGroup = CollisionGroup.TANK_TURRET_PARTS;
    options.interactsCollisionGroup = CollisionGroup.ALL | CollisionGroup.WALL | CollisionGroup.TANK_TURRET_PARTS | CollisionGroup.TANK_GUN_PARTS;

    // Создание турели
    updateColorOptions(options, [0.5, 1, 0.5, 1]);
    createRectanglesRR(turretEid, turretSet, options, 0 - 0.5 * PADDING, 0 - 8 * PADDING);

    // Настройка опций для орудия
    options.shadow[1] = 4;
    options.belongsCollisionGroup = CollisionGroup.TANK_GUN_PARTS;
    options.interactsCollisionGroup = CollisionGroup.BULLET | CollisionGroup.WALL | CollisionGroup.TANK_TURRET_PARTS | CollisionGroup.TANK_GUN_PARTS;

    // Создание орудия
    createRectanglesRR(turretEid, gunSet, options, 0 - 0.5 * PADDING, 0 - 8 * PADDING);
}

const parentVector = new Vector2(0, 0);
const childVector = new Vector2(0, 0);
const createRectanglesRR = (
    parentEId: number,
    params: [number, number, number, number][],
    options: Options,
    x: number,
    y: number,
    { world, physicalWorld } = GameDI,
) => {
    const rbId = RigidBodyRef.id[parentEId];
    const parentRb = physicalWorld.getRigidBody(rbId);

    childVector.x = 0;
    childVector.y = 0;

    return params.map((param) => {
        const parentTranslation = parentRb.translation();
        options.x = parentTranslation.x + x + param[0];
        options.y = parentTranslation.y + y + param[1];
        options.width = param[2];
        options.height = param[3];

        const [eid, pid] = createRectangleRR(options);

        parentVector.x = x + param[0];
        parentVector.y = y + param[1];
        const joint = physicalWorld.createImpulseJoint(
            JointData.fixed(parentVector, 0, childVector, 0),
            parentRb,
            physicalWorld.getRigidBody(pid),
            true,
        );

        Player.addComponent(world, eid, options.playerId);
        Hitable.addComponent(world, eid);
        Parent.addComponent(world, eid, parentEId);
        Children.addChildren(parentEId, eid);
        TankPart.addComponent(world, eid, joint.handle);

        return eid;
    });
};
