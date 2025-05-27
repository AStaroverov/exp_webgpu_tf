import { Color } from '../../../../../../../../src/ECS/Components/Common.ts';
import { Options } from './Options.ts';
import { GameDI } from '../../../../DI/GameDI.ts';
import { Tank } from '../../../Components/Tank.ts';
import { addTransformComponents } from '../../../../../../../../src/ECS/Components/Transform.ts';
import { Children } from '../../../Components/Children.ts';
import { TeamRef } from '../../../Components/TeamRef.ts';
import { PlayerRef } from '../../../Components/PlayerRef.ts';
import { TankController } from '../../../Components/TankController.ts';
import { HeuristicsData } from '../../../Components/HeuristicsData.ts';
import { TenserFlowDI } from '../../../../DI/TenserFlowDI.ts';
import { TankInputTensor } from '../../../Components/TankState.ts';
import { JointData, Vector2 } from '@dimforge/rapier2d-simd';
import { TankPart } from '../../../Components/TankPart.ts';
import { Parent } from '../../../Components/Parent.ts';
import { CollisionGroup } from '../../../../Physical/createRigid.ts';
import { createRectangleRigidGroup } from '../../../Components/RigidGroup.ts';
import { TankTurret } from '../../../Components/TankTurret.ts';

export function createTankBase(options: Options, { world } = GameDI): [number, number] {
    options.belongsCollisionGroup = CollisionGroup.TANK_BASE;
    options.interactsCollisionGroup = CollisionGroup.TANK_BASE;

    const [tankEid, tankPid] = createRectangleRigidGroup(options);
    // const [tankEid, tankPid] = createRectangleRR(options);
    Tank.addComponent(world, tankEid, options.tankType, options.partsCount);
    Tank.setEngineType(tankEid, options.engineType);
    Tank.setCaterpillarsLength(tankEid, options.caterpillarLength);

    // Добавление базовых компонентов
    addTransformComponents(world, tankEid);
    Children.addComponent(world, tankEid);
    TeamRef.addComponent(world, tankEid, options.teamId);
    PlayerRef.addComponent(world, tankEid, options.playerId);
    TankController.addComponent(world, tankEid);
    HeuristicsData.addComponent(world, tankEid, options.approximateColliderRadius);
    Color.addComponent(world, tankEid, ...options.color);

    // Добавление TensorFlow компонентов, если активировано
    if (TenserFlowDI.enabled) {
        TankInputTensor.addComponent(world, tankEid);
    }

    return [tankEid, tankPid];
}


const parentVector = new Vector2(0, 0);
const childVector = new Vector2(0, 0);

export function createTankTurret(options: Options, tankEid: number, tankPid: number, {
    world,
    physicalWorld,
} = GameDI): [number, number] {
    options.belongsCollisionGroup = 0;
    options.interactsCollisionGroup = 0;

    const [turretEid, turretPid] = createRectangleRigidGroup(options);
    // const [turretEid, turretPid] = createRectangleRR(options);
    TankTurret.addComponent(world, turretEid, tankEid);
    TankTurret.setReloadingDuration(turretEid, options.turret.reloadingDuration);
    TankTurret.setBulletData(turretEid, options.turret.bulletStartPosition, options.turret.bulletCaliber);
    TankTurret.setRotationSpeed(turretEid, options.turret.rotationSpeed);

    parentVector.x = 0;
    parentVector.y = 0;
    childVector.x = 0;
    childVector.y = 0;

    const joint = physicalWorld.createImpulseJoint(
        JointData.revolute(parentVector, childVector),
        physicalWorld.getRigidBody(tankPid),
        physicalWorld.getRigidBody(turretPid),
        false,
    );
    TankPart.addComponent(world, turretEid, joint.handle, parentVector, childVector);

    addTransformComponents(world, turretEid);
    Parent.addComponent(world, turretEid, tankEid);
    Children.addComponent(world, turretEid);

    Tank.setTurretEid(tankEid, turretEid);
    Children.addChildren(tankEid, turretEid);

    return [turretEid, turretPid];
}

