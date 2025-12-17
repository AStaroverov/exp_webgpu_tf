import { RigidBodyType } from '@dimforge/rapier2d-simd';
import { TColor } from '../../../../../../renderer/src/ECS/Components/Common.ts';
import { VehicleType } from '../../Components/Vehicle.ts';
import { VehicleEngineType } from '../../Systems/Vehicle/VehicleControllerSystems.ts';

export type VehicleOptions = typeof mutatedOptions;

export const mutatedOptions = {
    teamId: -1,
    playerId: -1,

    x: NaN,
    y: NaN,
    z: 0,
    width: 0,
    height: 0,
    radius: 0,
    rotation: NaN,
    speedX: 0,
    speedY: 0,
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

    vehicleType: VehicleType.LightTank,
    engineType: VehicleEngineType.v6,
};

export const defaultOptions = structuredClone(mutatedOptions);

export type VehicleCreationOpts = {
    playerId: number;
    teamId: number;
    x: number;
    y: number;
    rotation: number;
    color: TColor;
};

export const resetOptions = (target: VehicleOptions, source?: Partial<VehicleCreationOpts>) => {
    target.teamId = source?.teamId ?? defaultOptions.teamId;
    target.playerId = source?.playerId ?? defaultOptions.playerId;

    target.x = source?.x ?? defaultOptions.x;
    target.y = source?.y ?? defaultOptions.y;
    target.z = defaultOptions.z;
    target.width = defaultOptions.width;
    target.height = defaultOptions.height;
    target.radius = defaultOptions.radius;
    target.rotation = source?.rotation ?? defaultOptions.rotation;
    target.speedX = defaultOptions.speedX;
    target.speedY = defaultOptions.speedY;
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

    target.vehicleType = defaultOptions.vehicleType;
    target.engineType = defaultOptions.engineType;

    return target;
};

export const updateColorOptions = (target: VehicleOptions, color: TColor) => {
    (target.color as Float32Array).set(color, 0);
};

