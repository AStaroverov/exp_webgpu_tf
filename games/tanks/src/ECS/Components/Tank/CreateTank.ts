import { TColor } from '../../../../../../src/ECS/Components/Common.ts';
import { JointData, Vector2 } from '@dimforge/rapier2d-simd';
import { GameDI } from '../../../DI/GameDI.ts';
import { Player } from '../Player.ts';
import { Parent } from '../Parent.ts';
import { Children } from '../Children.ts';
import { TankPart } from './TankPart.ts';
import { CollisionGroup } from '../../../Physical/createRigid.ts';
import { createRectangleRigidGroup } from '../RigidGroup.ts';
import { addTransformComponents } from '../../../../../../src/ECS/Components/Transform.ts';
import { TankController } from './TankController.ts';
import { Team } from '../Team.ts';
import { TenserFlowDI } from '../../../DI/TenserFlowDI.ts';
import { TankInputTensor } from './TankState.ts';
import { createCircle } from '../../../../../../src/ECS/Entities/Shapes.ts';
import { Tank } from './Tank.ts';
import {
    createTankHullParts,
    createTankTracksParts,
    createTankTurretAndGunParts,
    PADDING,
    PARTS_COUNT,
} from './CreateTankParts.ts';
import { BASE_DENSITY, mutatedOptions, Options, resetOptions, updateColorOptions } from './Options.ts';

/**
 * Создает танк с его компонентами
 */
export function createTank(options: {
    playerId: number,
    teamId: number,
    x: number,
    y: number,
    rotation: number,
    color: TColor,
}) {
    // Инициализация опций
    const tankOptions = initTankOptions(options);

    // Создание базовой структуры танка
    const [tankEid, tankPid] = createTankBase(tankOptions);

    // Создание прицела
    createTankAim(tankOptions, tankEid);

    // Создание турели
    const [turretEid] = createTankTurret(tankOptions, tankEid, tankPid);

    // Создание корпуса
    createTankHullParts(tankOptions, tankEid, options.color);

    // Создание гусениц
    createTankTracksParts(tankOptions, tankEid);

    // Создание орудия на турели
    createTankTurretAndGunParts(tankOptions, turretEid);

    return tankEid;
}

/**
 * Инициализирует опции для создания танка
 */
function initTankOptions(options: {
    playerId: number,
    teamId: number,
    x: number,
    y: number,
    rotation: number,
    color: TColor,
}) {
    resetOptions(mutatedOptions, options);

    // Базовые параметры танка
    mutatedOptions.density = BASE_DENSITY * 10;
    mutatedOptions.width = PADDING * 12;
    mutatedOptions.height = PADDING * 14;
    mutatedOptions.belongsCollisionGroup = CollisionGroup.TANK_BASE;
    mutatedOptions.interactsCollisionGroup = CollisionGroup.TANK_BASE;

    return mutatedOptions;
}

/**
 * Создает базовую структуру танка и добавляет компоненты
 */
function createTankBase(options: Options, { world } = GameDI): [number, number] {
    const [tankEid, tankPid] = createRectangleRigidGroup(options);

    Tank.addComponent(
        world,
        tankEid,
        400,
        [1, -PADDING * 9],
        PARTS_COUNT,
    );

    // Добавление базовых компонентов
    addTransformComponents(world, tankEid);
    Children.addComponent(world, tankEid);
    Team.addComponent(world, tankEid, options.teamId);
    Player.addComponent(world, tankEid, options.playerId);
    TankController.addComponent(world, tankEid);

    // Добавление TensorFlow компонентов, если активировано
    if (TenserFlowDI.enabled) {
        TankInputTensor.addComponent(world, tankEid);
    }

    return [tankEid, tankPid];
}

/**
 * Создает прицел для танка
 */
function createTankAim(options: Options, tankEid: number, { world } = GameDI): number {
    options.radius = 16;
    const aimEid = createCircle(GameDI.world, options);

    // Связывание прицела с танком
    Tank.setAimEid(tankEid, aimEid);
    Parent.addComponent(world, aimEid, tankEid);
    Children.addChildren(tankEid, aimEid);

    return aimEid;
}

/**
 * Создает турель танка с шарниром
 */
const parentVector = new Vector2(0, 0);
const childVector = new Vector2(0, 0);

function createTankTurret(options: Options, tankEid: number, tankPid: number, {
    world,
    physicalWorld,
} = GameDI): [number, number] {
    // Настройка опций турели
    options.density = BASE_DENSITY;
    options.width = PADDING * 6;
    options.height = PADDING * 17;
    options.belongsCollisionGroup = 0;
    options.interactsCollisionGroup = 0;
    updateColorOptions(options, [0.5, 0, 0, 1]);

    // Создание турели
    const [turretEid, turretPid] = createRectangleRigidGroup(options);

    // Создание шарнира между турелью и базой
    parentVector.x = 0;
    parentVector.y = 0;
    childVector.x = 0;
    childVector.y = PADDING * 5;

    const joint = physicalWorld.createImpulseJoint(
        JointData.revolute(parentVector, childVector),
        physicalWorld.getRigidBody(tankPid),
        physicalWorld.getRigidBody(turretPid),
        true,
    );
    TankPart.addComponent(world, turretEid, joint.handle, parentVector, childVector);

    // Добавление компонентов турели
    addTransformComponents(world, turretEid);
    Parent.addComponent(world, turretEid, tankEid);
    Children.addComponent(world, turretEid);

    // Связывание турели с танком
    Tank.setTurretEid(tankEid, turretEid);
    Children.addChildren(tankEid, turretEid);

    return [turretEid, turretPid];
}

