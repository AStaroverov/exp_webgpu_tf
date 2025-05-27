import { RigidBodyType } from '@dimforge/rapier2d-simd';
import { TColor } from '../../../../../../../../src/ECS/Components/Common.ts';
import { createMediumTank } from '../Medium/MediumTank.ts';
import { TankEngineType } from '../../../Systems/Tank/TankControllerSystems.ts';
import { BulletCaliber } from '../../../Components/Bullet.ts';
import { TankType } from '../../../Components/Tank.ts';

export type Options = typeof mutatedOptions;

export const mutatedOptions = {
    teamId: -1,
    playerId: -1,

    x: 0,
    y: 0,
    z: 0,
    width: 0,
    height: 0,
    radius: 0,
    rotation: 0,
    color: new Float32Array([1, 0, 0, 1]),
    shadow: new Float32Array([0, 3]),
    bodyType: RigidBodyType.Dynamic,
    density: 0,
    linearDamping: 5,
    angularDamping: 5,
    belongsSolverGroup: 0,
    interactsSolverGroup: 0,
    belongsCollisionGroup: 0,
    interactsCollisionGroup: 0,

    partsCount: 0,
    size: 0,
    padding: 0,
    approximateColliderRadius: 0,

    tankType: TankType.Light,
    engineType: TankEngineType.v6,
    caterpillarLength: 0,

    turret: {
        rotationSpeed: 0,
        reloadingDuration: 0,
        bulletStartPosition: [0, 0],
        bulletCaliber: BulletCaliber.Light,
    },
};

export const defaultOptions = structuredClone(mutatedOptions);
export const resetOptions = (target: Options, source: Parameters<typeof createMediumTank>[0]) => {
    target.teamId = source?.teamId ?? defaultOptions.teamId;
    target.playerId = source?.playerId ?? defaultOptions.playerId;

    target.x = source?.x ?? defaultOptions.x;
    target.y = source?.y ?? defaultOptions.y;
    target.z = defaultOptions.z;
    target.width = defaultOptions.width;
    target.height = defaultOptions.height;
    target.radius = defaultOptions.radius;
    target.rotation = source?.rotation ?? defaultOptions.rotation;
    (target.color as Float32Array).set(source?.color ?? defaultOptions.color, 0);
    (target.shadow as Float32Array).set(defaultOptions.shadow, 0);
    target.density = defaultOptions.density;
    target.linearDamping = defaultOptions.linearDamping;
    target.angularDamping = defaultOptions.angularDamping;
    target.belongsSolverGroup = defaultOptions.belongsSolverGroup;
    target.interactsSolverGroup = defaultOptions.interactsSolverGroup;
    target.belongsCollisionGroup = defaultOptions.belongsCollisionGroup;
    target.interactsCollisionGroup = defaultOptions.interactsCollisionGroup;

    target.partsCount = defaultOptions.partsCount;
    target.size = defaultOptions.size;
    target.padding = defaultOptions.padding;
    target.approximateColliderRadius = defaultOptions.approximateColliderRadius;

    target.tankType = defaultOptions.tankType;
    target.engineType = defaultOptions.engineType;
    target.caterpillarLength = defaultOptions.caterpillarLength;

    target.turret.bulletStartPosition = defaultOptions.turret.bulletStartPosition;
    target.turret.bulletCaliber = defaultOptions.turret.bulletCaliber;

    return target;
};

export const updateColorOptions = (target: Options, color: TColor) => {
    (target.color as Float32Array).set(color, 0);
};
