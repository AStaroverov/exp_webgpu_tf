import { query, hasComponent } from 'bitecs';
import { Vector2 } from '@dimforge/rapier2d-simd';
import { GameDI } from '../../../DI/GameDI.ts';
import { RigidBodyState } from '../../Components/Physical.ts';
import { Track, TrackSide } from '../../Components/Track.ts';
import { VehicleController } from '../../Components/VehicleController.ts';
import { Vehicle } from '../../Components/Vehicle.ts';
import { applyRotationToVector } from '../../../Physical/applyRotationToVector.ts';
import { Impulse } from '../../Components/Impulse.ts';
import { Children } from '../../Components/Children.ts';
import { Tank } from '../../Components/Tank.ts';
import { EngineType } from '../../../Config/vehicles.ts';

const TRACK_IMPULSE_FACTOR = 3_000_000_000; // Half of original since we have 2 tracks

const mapTypeToTrackImpulse = {
    [EngineType.v6]: TRACK_IMPULSE_FACTOR * 0.8,
    [EngineType.v8]: TRACK_IMPULSE_FACTOR,
    [EngineType.v12]: TRACK_IMPULSE_FACTOR * 2,
    [EngineType.v8_turbo]: TRACK_IMPULSE_FACTOR * 2,
};

const impulseVector = new Vector2(0, 0);

export function createTrackControlSystem({ world } = GameDI) {
    return (delta: number) => {
        const vehicleEids = query(world, [Tank, Vehicle, VehicleController, Children]);

        for (let i = 0; i < vehicleEids.length; i++) {
            const vehicleEid = vehicleEids[i];
            const moveDirection = VehicleController.move[vehicleEid];
            const rotationDirection = VehicleController.rotation[vehicleEid];
            
            const engineType = Vehicle.engineType[vehicleEid] as EngineType;
            const impulseFactor = mapTypeToTrackImpulse[engineType];
            const vehicleRotation = RigidBodyState.rotation[vehicleEid];

            // Calculate differential track powers for turning
            // When turning right (rotationDirection > 0): left track faster, right track slower
            // When turning left (rotationDirection < 0): right track faster, left track slower
            const turnFactor = -0.7; // How much rotation affects track differential
            
            // Base power from movement direction
            let leftPower = moveDirection;
            let rightPower = moveDirection;
            
            // Add rotation differential
            // For rotation-only (stationary turn): one track forward, one backward
            // For movement + rotation: differential speed
            leftPower += rotationDirection * turnFactor;
            rightPower -= rotationDirection * turnFactor;
            
            // Normalize if either power exceeds 1
            const maxPower = Math.max(Math.abs(leftPower), Math.abs(rightPower));
            if (maxPower > 1) {
                leftPower /= maxPower;
                rightPower /= maxPower;
            }

            const childCount = Children.entitiesCount[vehicleEid];
            
            // Process each child that is a track
            for (let c = 0; c < childCount; c++) {
                const childEid = Children.entitiesIds.get(vehicleEid, c);
                
                if (!hasComponent(world, childEid, Track)) {
                    continue;
                }
                
                const trackSide = Track.side[childEid];
                const power = trackSide === TrackSide.Left ? leftPower : rightPower;
                applyTrackImpulse(childEid, power * impulseFactor, vehicleRotation, delta);
            }
        }
    };
}

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

