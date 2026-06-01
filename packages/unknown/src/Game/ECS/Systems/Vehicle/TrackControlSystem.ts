import { query, hasComponent } from 'bitecs';
import { Vector2 } from '@dimforge/rapier2d-simd';
import { TrackSide } from '../../Components/Track.ts';
import { applyRotationToVector } from '../../../Physical/applyRotationToVector.ts';
import { EngineType } from '../../../Config/vehicles.ts';
import { getPhysicsWorldComponents } from '../../createPhysicsWorld.ts';
import { getBrainWorldComponents } from '../../createBrainWorld.ts';
import { getNodeChildren, getNodePhysics } from '../../refs.ts';
import { Worlds } from '../../../DI/Worlds.ts';

const TRACK_IMPULSE_FACTOR = 3_000_000_000;

const mapTypeToTrackImpulse = {
    [EngineType.v6]: TRACK_IMPULSE_FACTOR * 0.8,
    [EngineType.v8]: TRACK_IMPULSE_FACTOR,
    [EngineType.v12]: TRACK_IMPULSE_FACTOR * 2,
    [EngineType.v8_turbo]: TRACK_IMPULSE_FACTOR * 2,
};

const impulseVector = new Vector2(0, 0);

export function createTrackControlSystem({ physicsWorld, brainWorld } = Worlds) {
    const { Track, RigidBodyState, Impulse } = getPhysicsWorldComponents(physicsWorld);
    const { Tank, Vehicle, VehicleController } = getBrainWorldComponents(brainWorld);

    function applyTrackImpulse(
        trackEid: number,
        impulseFactor: number,
        vehicleRotation: number,
        delta: number,
    ) {
        if (impulseFactor === 0) return;
        impulseVector.x = impulseFactor * delta / 1000;
        impulseVector.y = 0;
        applyRotationToVector(impulseVector, impulseVector, vehicleRotation);
        Impulse.add(trackEid, impulseVector.x, impulseVector.y);
    }

    return (delta: number) => {
        const brainEids = query(brainWorld, [Tank, Vehicle, VehicleController]);

        for (let i = 0; i < brainEids.length; i++) {
            const brainEid = brainEids[i];
            // brainEid IS the hull node; its presentation (downward) is the hull atom,
            // and its Brain children are the turret/track/wheel nodes.
            const vehicleEid = getNodePhysics(brainEid);
            const childNodes = getNodeChildren(brainEid);
            if (childNodes.length === 0) continue;
            const moveDirection = VehicleController.move[brainEid];
            const rotationDirection = VehicleController.rotation[brainEid];

            const engineType = Vehicle.engineType[brainEid] as EngineType;
            const impulseFactor = mapTypeToTrackImpulse[engineType];
            const vehicleRotation = RigidBodyState.rotation[vehicleEid];

            const turnFactor = -0.7;

            let leftPower = moveDirection;
            let rightPower = moveDirection;

            leftPower += rotationDirection * turnFactor;
            rightPower -= rotationDirection * turnFactor;

            const maxPower = Math.max(Math.abs(leftPower), Math.abs(rightPower));
            if (maxPower > 1) {
                leftPower /= maxPower;
                rightPower /= maxPower;
            }

            for (let c = 0; c < childNodes.length; c++) {
                const childEid = getNodePhysics(childNodes[c]);
                if (childEid === 0) continue;

                if (!hasComponent(physicsWorld, childEid, Track)) {
                    continue;
                }

                const trackSide = Track.side[childEid];
                const power = trackSide === TrackSide.Left ? leftPower : rightPower;
                applyTrackImpulse(childEid, power * impulseFactor, vehicleRotation, delta);
            }
        }
    };
}
