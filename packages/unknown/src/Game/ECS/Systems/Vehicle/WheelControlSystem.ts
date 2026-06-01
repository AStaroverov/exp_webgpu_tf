import { query, hasComponent } from 'bitecs';
import { Vector2 } from '@dimforge/rapier2d-simd';
import { applyRotationToVector } from '../../../Physical/applyRotationToVector.ts';
import { clamp } from 'lodash-es';
import { EngineType } from '../../../Config/vehicles.ts';
import { getPhysicsWorldComponents } from '../../createPhysicsWorld.ts';
import { getBrainWorldComponents } from '../../createBrainWorld.ts';
import { getNodeChildren, getNodePhysics } from '../../refs.ts';
import { Worlds } from '../../../DI/Worlds.ts';

const WHEEL_IMPULSE_FACTOR = 4000000000;

const mapTypeToWheelImpulse = {
    [EngineType.v6]: WHEEL_IMPULSE_FACTOR * 0.8,
    [EngineType.v8]: WHEEL_IMPULSE_FACTOR,
    [EngineType.v12]: WHEEL_IMPULSE_FACTOR * 2,
    [EngineType.v8_turbo]: WHEEL_IMPULSE_FACTOR * 2,
};

const impulseVector = new Vector2(0, 0);

export function createWheelControlSystem({ physicsWorld, brainWorld } = Worlds) {
    const {
        Wheel, WheelDrive, WheelSteerable,
        JointMotor, Impulse, RigidBodyState,
    } = getPhysicsWorldComponents(physicsWorld);
    const { Vehicle, VehicleController } = getBrainWorldComponents(brainWorld);

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
        const brainEids = query(brainWorld, [Vehicle, VehicleController]);

        for (let i = 0; i < brainEids.length; i++) {
            const brainEid = brainEids[i];
            // brainEid IS the hull node; its Brain children are the turret/track/wheel nodes.
            const childNodes = getNodeChildren(brainEid);
            if (childNodes.length === 0) continue;

            const accelerate = VehicleController.move[brainEid];
            const steering = VehicleController.rotation[brainEid];

            const engineType = Vehicle.engineType[brainEid] as EngineType;
            const impulseFactor = mapTypeToWheelImpulse[engineType];

            for (let c = 0; c < childNodes.length; c++) {
                const childEid = getNodePhysics(childNodes[c]);
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
