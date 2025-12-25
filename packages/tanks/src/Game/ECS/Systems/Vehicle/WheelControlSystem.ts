import { query, hasComponent } from 'bitecs';
import { Vector2 } from '@dimforge/rapier2d-simd';
import { GameDI } from '../../../DI/GameDI.ts';
import { RigidBodyState } from '../../Components/Physical.ts';
import { Wheel, WheelDrive, WheelSteerable } from '../../Components/Wheel.ts';
import { VehicleController } from '../../Components/VehicleController.ts';
import { Vehicle } from '../../Components/Vehicle.ts';
import { VehicleEngineType } from './VehicleControllerSystems.ts';
import { applyRotationToVector } from '../../../Physical/applyRotationToVector.ts';
import { Impulse } from '../../Components/Impulse.ts';
import { Children } from '../../Components/Children.ts';
import { JointMotor } from '../../Components/JointMotor.ts';
import { clamp } from 'lodash-es';

const WHEEL_IMPULSE_FACTOR = 4000000000; // Per wheel

const mapTypeToWheelImpulse = {
    [VehicleEngineType.v6]: WHEEL_IMPULSE_FACTOR * 0.8,
    [VehicleEngineType.v8]: WHEEL_IMPULSE_FACTOR,
    [VehicleEngineType.v12]: WHEEL_IMPULSE_FACTOR * 2,
    [VehicleEngineType.v8_turbo]: WHEEL_IMPULSE_FACTOR * 2,
};

const impulseVector = new Vector2(0, 0);

/**
 * System that controls wheeled vehicles.
 * - Steerable wheels (typically front) rotate based on steering input
 * - Drive wheels receive power impulse based on acceleration input
 * 
 * Uses Ackermann-like steering where front wheels turn to steer the vehicle.
 */
export function createWheelControlSystem({ world } = GameDI) {
    return (delta: number) => {
        const vehicleEids = query(world, [Vehicle, VehicleController, Children]);

        for (let i = 0; i < vehicleEids.length; i++) {
            const vehicleEid = vehicleEids[i];
            const accelerate = VehicleController.move[vehicleEid];
            const steering = VehicleController.rotation[vehicleEid];
            
            const engineType = Vehicle.engineType[vehicleEid] as VehicleEngineType;
            const impulseFactor = mapTypeToWheelImpulse[engineType];

            const childCount = Children.entitiesCount[vehicleEid];
            
            // Process each child that is a wheel
            for (let c = 0; c < childCount; c++) {
                const childEid = Children.entitiesIds.get(vehicleEid, c);
                
                if (!hasComponent(world, childEid, Wheel)) {
                    continue;
                }
                
                // Handle steering for steerable wheels
                if (hasComponent(world, childEid, WheelSteerable)) {
                    const maxAngle = WheelSteerable.maxSteeringAngle[childEid];
                    const targetAngle = clamp(steering * maxAngle, -maxAngle, maxAngle);
                    JointMotor.setTargetPosition$(childEid, targetAngle);
                }
                
                // Handle drive for drive wheels
                if (hasComponent(world, childEid, WheelDrive)) {
                    applyWheelDrive(childEid, accelerate, impulseFactor, delta);
                }
            }
        }
    };
}

/**
 * Applies drive impulse at the wheel's world position.
 * Force applied off-center creates natural steering behavior.
 */
function applyWheelDrive(
    wheelEid: number,
    accelerate: number,
    impulseFactor: number,
    delta: number,
) {
    if (accelerate === 0) return;
    
    const steeringAngle = RigidBodyState.rotation[wheelEid];    
    impulseVector.x = accelerate * impulseFactor * delta / 1000;
    impulseVector.y = 0;
    
    applyRotationToVector(impulseVector, impulseVector, steeringAngle);

    // Apply impulse at wheel position for realistic physics
    Impulse.add(wheelEid, impulseVector.x, impulseVector.y);
}

/**
 * Creates a simpler wheel query system that checks drive wheels.
 */
export function createWheelDriveQuery({ world } = GameDI) {
    const driveWheels = query(world, [Wheel, WheelDrive]);
    return () => driveWheels;
}

