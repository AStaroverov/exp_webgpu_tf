import { RigidBodyType } from '@dimforge/rapier2d-simd';
import { TColor } from '../../../../../../renderer/src/ECS/Components/Common.ts';
import { VehicleType } from '../../Components/Vehicle.ts';
import { VehicleEngineType } from '../../Systems/Vehicle/VehicleControllerSystems.ts';

export type VehicleOptions = typeof mutatedVehicleOptions;

export const mutatedVehicleOptions = {
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

export const defaultVehicleOptions = Object.freeze(structuredClone(mutatedVehicleOptions));

export type VehicleCreationOpts = {
    playerId: number;
    teamId: number;
    x: number;
    y: number;
    rotation: number;
    color: TColor;
};

export const resetOptions = (target: VehicleOptions, source?: Partial<VehicleCreationOpts>) => {
    target.teamId = source?.teamId ?? defaultVehicleOptions.teamId;
    target.playerId = source?.playerId ?? defaultVehicleOptions.playerId;

    target.x = source?.x ?? defaultVehicleOptions.x;
    target.y = source?.y ?? defaultVehicleOptions.y;
    target.z = defaultVehicleOptions.z;
    target.width = defaultVehicleOptions.width;
    target.height = defaultVehicleOptions.height;
    target.radius = defaultVehicleOptions.radius;
    target.rotation = source?.rotation ?? defaultVehicleOptions.rotation;
    target.speedX = defaultVehicleOptions.speedX;
    target.speedY = defaultVehicleOptions.speedY;
    (target.color as Float32Array).set(source?.color ?? defaultVehicleOptions.color, 0);
    (target.shadow as Float32Array).set(defaultVehicleOptions.shadow, 0);
    target.density = defaultVehicleOptions.density;
    target.linearDamping = defaultVehicleOptions.linearDamping;
    target.angularDamping = defaultVehicleOptions.angularDamping;
    target.belongsSolverGroup = defaultVehicleOptions.belongsSolverGroup;
    target.interactsSolverGroup = defaultVehicleOptions.interactsSolverGroup;
    target.belongsCollisionGroup = defaultVehicleOptions.belongsCollisionGroup;
    target.interactsCollisionGroup = defaultVehicleOptions.interactsCollisionGroup;

    target.partsCount = defaultVehicleOptions.partsCount;
    target.size = defaultVehicleOptions.size;
    target.padding = defaultVehicleOptions.padding;
    target.approximateColliderRadius = defaultVehicleOptions.approximateColliderRadius;

    target.vehicleType = defaultVehicleOptions.vehicleType;
    target.engineType = defaultVehicleOptions.engineType;

    return target;
};

export const updateColorOptions = (target: VehicleOptions, color: TColor) => {
    (target.color as Float32Array).set(color, 0);
};

