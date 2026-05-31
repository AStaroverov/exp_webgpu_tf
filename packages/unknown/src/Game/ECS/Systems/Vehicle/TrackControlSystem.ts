import { query, hasComponent } from 'bitecs';
import { Vector2 } from '@dimforge/rapier2d-simd';
import { TrackSide } from '../../Components/Track.ts';
import { applyRotationToVector } from '../../../Physical/applyRotationToVector.ts';
import { EngineType } from '../../../Config/vehicles.ts';
import { getPhysicsWorldComponents } from '../../createPhysicsWorld.ts';
import { getRenderWorldComponents } from '../../createRenderWorld.ts';
import { BridgeDI } from '../../../DI/BridgeDI.ts';
import { Worlds } from '../../../DI/Worlds.ts';

const TRACK_IMPULSE_FACTOR = 3_000_000_000;

const mapTypeToTrackImpulse = {
    [EngineType.v6]: TRACK_IMPULSE_FACTOR * 0.8,
    [EngineType.v8]: TRACK_IMPULSE_FACTOR,
    [EngineType.v12]: TRACK_IMPULSE_FACTOR * 2,
    [EngineType.v8_turbo]: TRACK_IMPULSE_FACTOR * 2,
};

const impulseVector = new Vector2(0, 0);

export function createTrackControlSystem({ physicsWorld, renderWorld } = Worlds) {
    const { Tank, Vehicle, VehicleController, Track, RigidBodyState, Impulse } = getPhysicsWorldComponents(physicsWorld);

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
        const { Children } = getRenderWorldComponents(renderWorld);
        const vehicleEids = query(physicsWorld, [Tank, Vehicle, VehicleController]);

        for (let i = 0; i < vehicleEids.length; i++) {
            const vehicleEid = vehicleEids[i];
            const vehicleRenderEid = BridgeDI.getRenderOf(vehicleEid);
            if (!hasComponent(renderWorld, vehicleRenderEid, Children)) continue;
            const moveDirection = VehicleController.move[vehicleEid];
            const rotationDirection = VehicleController.rotation[vehicleEid];

            const engineType = Vehicle.engineType[vehicleEid] as EngineType;
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

            const childCount = Children.entitiesCount[vehicleRenderEid];

            for (let c = 0; c < childCount; c++) {
                const childRenderEid = Children.entitiesIds.get(vehicleRenderEid, c);
                const childEid = BridgeDI.getPhysicsOf(childRenderEid);
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
