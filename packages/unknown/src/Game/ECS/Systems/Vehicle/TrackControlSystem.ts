import { query, hasComponent } from 'bitecs';
import { Vector2 } from '@dimforge/rapier2d-simd';
import { GameDI } from '../../../DI/GameDI.ts';
import { TrackSide } from '../../Components/Track.ts';
import { applyRotationToVector } from '../../../Physical/applyRotationToVector.ts';
import { EngineConfig, EngineType } from '../../../Config/vehicles.ts';
import { getGameComponents } from '../../createGameWorld.ts';

const TRACK_IMPULSE_FACTOR = 3_000_000_000;

const mapTypeToTrackImpulse = {
    [EngineType.v6]: TRACK_IMPULSE_FACTOR * EngineConfig[EngineType.v6].impulseMult,
    [EngineType.v8]: TRACK_IMPULSE_FACTOR * EngineConfig[EngineType.v8].impulseMult,
    [EngineType.v12]: TRACK_IMPULSE_FACTOR * EngineConfig[EngineType.v12].impulseMult,
};

const impulseVector = new Vector2(0, 0);

export function createTrackControlSystem({ world } = GameDI) {
    const { Tank, Vehicle, VehicleController, Children, Track, RigidBodyState, Impulse, Slowed } = getGameComponents(world);

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
        const vehicleEids = query(world, [Tank, Vehicle, VehicleController, Children]);

        for (let i = 0; i < vehicleEids.length; i++) {
            const vehicleEid = vehicleEids[i];
            const moveDirection = VehicleController.move[vehicleEid];
            const rotationDirection = VehicleController.rotation[vehicleEid];

            const engineType = Vehicle.engineType[vehicleEid] as EngineType;
            const impulseFactor = mapTypeToTrackImpulse[engineType];
            const vehicleRotation = RigidBodyState.rotation[vehicleEid];
            const slow = hasComponent(world, vehicleEid, Slowed) ? 1 - Slowed.slowMul[vehicleEid] : 1;

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

            const childCount = Children.entitiesCount[vehicleEid];

            for (let c = 0; c < childCount; c++) {
                const childEid = Children.entitiesIds.get(vehicleEid, c);

                if (!hasComponent(world, childEid, Track)) {
                    continue;
                }

                const trackSide = Track.side[childEid];
                const power = trackSide === TrackSide.Left ? leftPower : rightPower;
                applyTrackImpulse(childEid, power * impulseFactor * slow, vehicleRotation, delta);
            }
        }
    };
}
