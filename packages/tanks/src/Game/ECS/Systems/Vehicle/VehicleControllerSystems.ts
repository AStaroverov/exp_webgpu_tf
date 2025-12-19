import { GameDI } from '../../../DI/GameDI.ts';
import { TurretController } from '../../Components/TurretController.ts';
import { VehicleController } from '../../Components/VehicleController.ts';
import { Vector2 } from '@dimforge/rapier2d-simd';
import { Vehicle } from '../../Components/Vehicle.ts';
import { RigidBodyState } from '../../Components/Physical.ts';
import { applyRotationToVector } from '../../../Physical/applyRotationToVector.ts';
import { query } from 'bitecs';
import { normalizeAngle } from '../../../../../../../lib/math.ts';
import { VehicleTurret } from '../../Components/VehicleTurret.ts';
import { Parent } from '../../Components/Parent.ts';
import { Impulse, TorqueImpulse } from '../../Components/Impulse.ts';
import { JointMotor } from '../../Components/JointMotor.ts';

export enum VehicleEngineType {
    v6,
    v8,
    v12,
    v8_turbo,  // Player special engine - faster than v8
}

export const mapVehicleEngineLabel = {
    [VehicleEngineType.v6]: 'v6',
    [VehicleEngineType.v8]: 'v8',
    [VehicleEngineType.v12]: 'v12',
    [VehicleEngineType.v8_turbo]: 'v8 Turbo',
};

const IMPULSE_FACTOR = 15000000000;
const ROTATION_IMPULSE_FACTOR = 150000000000;
const mapTypeToFeatures = {
    [VehicleEngineType.v6]: {
        impulseFactor: IMPULSE_FACTOR * 0.8,
        rotationImpulseFactor: ROTATION_IMPULSE_FACTOR * 0.9,
    },
    [VehicleEngineType.v8]: {
        impulseFactor: IMPULSE_FACTOR,
        rotationImpulseFactor: ROTATION_IMPULSE_FACTOR,
    },
    [VehicleEngineType.v12]: {
        impulseFactor: IMPULSE_FACTOR * 2,
        rotationImpulseFactor: ROTATION_IMPULSE_FACTOR * 3,
    },
    [VehicleEngineType.v8_turbo]: {
        impulseFactor: IMPULSE_FACTOR * 2,
        rotationImpulseFactor: ROTATION_IMPULSE_FACTOR * 2,
    },
};

const nextLinvel = new Vector2(0, 0);

export function createVehiclePositionSystem({ world } = GameDI) {
    return (delta: number) => {
        const vehicleEids = query(world, [Vehicle, VehicleController]);

        for (let i = 0; i < vehicleEids.length; i++) {
            const vehicleEid = vehicleEids[i];
            const moveDirection = VehicleController.move[vehicleEid];
            const rotationDirection = VehicleController.rotation[vehicleEid];
            const {
                impulseFactor,
                rotationImpulseFactor,
            } = mapTypeToFeatures[Vehicle.engineType[vehicleEid] as VehicleEngineType];

            if (moveDirection === 0 && rotationDirection === 0) continue;

            const rotation = RigidBodyState.rotation[vehicleEid];
            // Задаем направление движения
            nextLinvel.x = 0;
            nextLinvel.y = -moveDirection * impulseFactor * delta / 1000;
            applyRotationToVector(nextLinvel, nextLinvel, rotation);

            // Add linear impulse for movement (will be applied by system)
            Impulse.add(vehicleEid, nextLinvel.x, nextLinvel.y);
            // Add torque impulse for rotation (will be applied by system)
            TorqueImpulse.add(vehicleEid, rotationDirection * rotationImpulseFactor * delta / 1000);
        }
    };
}

export function createVehicleTurretRotationSystem({ world } = GameDI) {
    return (delta: number) => {
        const turretEids = query(world, [VehicleTurret, TurretController, JointMotor]);

        for (let i = 0; i < turretEids.length; i++) {
            const turretEid = turretEids[i];
            const vehicleEid = Parent.id[turretEid];
            const vehicleRot = RigidBodyState.rotation[vehicleEid];
            const turretRot = RigidBodyState.rotation[turretEid];
            const turretRotDir = TurretController.rotation[turretEid];
            const maxRotationSpeed = VehicleTurret.rotationSpeed[turretEid];

            const relTurretRot = normalizeAngle(turretRot - vehicleRot);
            const deltaRot = turretRotDir * maxRotationSpeed * (delta / 1000);

            JointMotor.setTargetPosition$(turretEid, normalizeAngle(relTurretRot + deltaRot));
        }
    };
}

