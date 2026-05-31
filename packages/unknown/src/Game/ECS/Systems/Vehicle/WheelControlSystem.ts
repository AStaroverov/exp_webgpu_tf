import { query, hasComponent } from 'bitecs';
import { Vector2 } from '@dimforge/rapier2d-simd';
import { applyRotationToVector } from '../../../Physical/applyRotationToVector.ts';
import { clamp } from 'lodash-es';
import { EngineType } from '../../../Config/vehicles.ts';
import { getPhysicsWorldComponents } from '../../createPhysicsWorld.ts';
import { getRenderWorldComponents } from '../../createRenderWorld.ts';
import { BridgeDI } from '../../../DI/BridgeDI.ts';
import { Worlds } from '../../../DI/Worlds.ts';

const WHEEL_IMPULSE_FACTOR = 4000000000;

const mapTypeToWheelImpulse = {
    [EngineType.v6]: WHEEL_IMPULSE_FACTOR * 0.8,
    [EngineType.v8]: WHEEL_IMPULSE_FACTOR,
    [EngineType.v12]: WHEEL_IMPULSE_FACTOR * 2,
    [EngineType.v8_turbo]: WHEEL_IMPULSE_FACTOR * 2,
};

const impulseVector = new Vector2(0, 0);

export function createWheelControlSystem({ physicsWorld, renderWorld } = Worlds) {
    const {
        Vehicle, VehicleController, Wheel, WheelDrive, WheelSteerable,
        JointMotor, Impulse, RigidBodyState,
    } = getPhysicsWorldComponents(physicsWorld);

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
        const { Children } = getRenderWorldComponents(renderWorld);
        const vehicleEids = query(physicsWorld, [Vehicle, VehicleController]);

        for (let i = 0; i < vehicleEids.length; i++) {
            const vehicleEid = vehicleEids[i];
            const vehicleRenderEid = BridgeDI.getRenderOf(vehicleEid);
            if (!hasComponent(renderWorld, vehicleRenderEid, Children)) continue;

            const accelerate = VehicleController.move[vehicleEid];
            const steering = VehicleController.rotation[vehicleEid];

            const engineType = Vehicle.engineType[vehicleEid] as EngineType;
            const impulseFactor = mapTypeToWheelImpulse[engineType];

            const childCount = Children.entitiesCount[vehicleRenderEid];

            for (let c = 0; c < childCount; c++) {
                const childRenderEid = Children.entitiesIds.get(vehicleRenderEid, c);
                const childEid = BridgeDI.getPhysicsOf(childRenderEid);
                if (childEid === 0) continue;

                if (!hasComponent(physicsWorld, childEid, Wheel)) {
                    continue;
                }

                if (hasComponent(physicsWorld, childEid, WheelSteerable)) {
                    const maxAngle = WheelSteerable.maxSteeringAngle[childEid];
                    const targetAngle = clamp(steering * maxAngle, -maxAngle, maxAngle);
                    JointMotor.setTargetPosition$(childEid, targetAngle);
                }

                if (hasComponent(physicsWorld, childEid, WheelDrive)) {
                    applyWheelDrive(childEid, accelerate, impulseFactor, delta);
                }
            }
        }
    };
}
