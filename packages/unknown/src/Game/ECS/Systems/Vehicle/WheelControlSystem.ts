import { query, hasComponent } from 'bitecs';
import { Vector2 } from '@dimforge/rapier2d-simd';
import { GameDI } from '../../../DI/GameDI.ts';
import { applyRotationToVector } from '../../../Physical/applyRotationToVector.ts';
import { clamp } from 'lodash-es';
import { EngineType } from '../../../Config/vehicles.ts';
import { getGameComponents } from '../../createGameWorld.ts';

const WHEEL_IMPULSE_FACTOR = 4000000000;

const mapTypeToWheelImpulse = {
    [EngineType.v6]: WHEEL_IMPULSE_FACTOR * 0.8,
    [EngineType.v8]: WHEEL_IMPULSE_FACTOR,
    [EngineType.v12]: WHEEL_IMPULSE_FACTOR * 2,
    [EngineType.v8_turbo]: WHEEL_IMPULSE_FACTOR * 2,
};

const impulseVector = new Vector2(0, 0);

export function createWheelControlSystem({ world } = GameDI) {
    const {
        Vehicle, VehicleController, Children, Wheel, WheelDrive, WheelSteerable,
        JointMotor, Impulse, RigidBodyState,
    } = getGameComponents(world);

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

        Impulse.add(wheelEid, impulseVector.x, impulseVector.y);
    }

    return (delta: number) => {
        const vehicleEids = query(world, [Vehicle, VehicleController, Children]);

        for (let i = 0; i < vehicleEids.length; i++) {
            const vehicleEid = vehicleEids[i];
            const accelerate = VehicleController.move[vehicleEid];
            const steering = VehicleController.rotation[vehicleEid];

            const engineType = Vehicle.engineType[vehicleEid] as EngineType;
            const impulseFactor = mapTypeToWheelImpulse[engineType];

            const childCount = Children.entitiesCount[vehicleEid];

            for (let c = 0; c < childCount; c++) {
                const childEid = Children.entitiesIds.get(vehicleEid, c);

                if (!hasComponent(world, childEid, Wheel)) {
                    continue;
                }

                if (hasComponent(world, childEid, WheelSteerable)) {
                    const maxAngle = WheelSteerable.maxSteeringAngle[childEid];
                    const targetAngle = clamp(steering * maxAngle, -maxAngle, maxAngle);
                    JointMotor.setTargetPosition$(childEid, targetAngle);
                }

                if (hasComponent(world, childEid, WheelDrive)) {
                    applyWheelDrive(childEid, accelerate, impulseFactor, delta);
                }
            }
        }
    };
}
