import { query, hasComponent } from 'bitecs';
import { Vector2 } from '@dimforge/rapier2d-simd';
import { GameDI } from '../../../DI/GameDI.ts';
import { RigidBodyState } from '../../Components/Physical.ts';
import { Track, TrackSide } from '../../Components/Track.ts';
import { VehicleController } from '../../Components/VehicleController.ts';
import { Vehicle } from '../../Components/Vehicle.ts';
import { VehicleEngineType } from './VehicleControllerSystems.ts';
import { applyRotationToVector } from '../../../Physical/applyRotationToVector.ts';
import { Impulse } from '../../Components/Impulse.ts';
import { Children } from '../../Components/Children.ts';

const TRACK_IMPULSE_FACTOR = 7500000000; // Half of original since we have 2 tracks

const mapTypeToTrackImpulse = {
    [VehicleEngineType.v6]: TRACK_IMPULSE_FACTOR * 0.8,
    [VehicleEngineType.v8]: TRACK_IMPULSE_FACTOR,
    [VehicleEngineType.v12]: TRACK_IMPULSE_FACTOR * 2,
    [VehicleEngineType.v8_turbo]: TRACK_IMPULSE_FACTOR * 2,
};

const impulseVector = new Vector2(0, 0);

/**
 * System that controls tracked vehicles.
 * Translates vehicle movement/rotation input into individual track powers.
 * 
 * Each track applies force at its physical position on the vehicle.
 * When tracks have different powers, this creates natural rotation
 * because forces are applied at different points relative to center of mass.
 * 
 * Movement: Both tracks get equal power → vehicle moves straight
 * Rotation: Differential power → asymmetric forces → natural torque
 */
export function createTrackControlSystem({ world } = GameDI) {
    return (delta: number) => {
        const vehicleEids = query(world, [Vehicle, VehicleController, Children]);

        for (let i = 0; i < vehicleEids.length; i++) {
            const vehicleEid = vehicleEids[i];
            const moveDirection = VehicleController.move[vehicleEid];
            const rotationDirection = VehicleController.rotation[vehicleEid];
            
            const engineType = Vehicle.engineType[vehicleEid] as VehicleEngineType;
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

/**
 * Apply impulse to vehicle at the track's world position.
 * Force applied off-center creates both linear and angular acceleration.
 */
function applyTrackImpulse(
    trackEid: number,
    impulseFactor: number,
    vehicleRotation: number,
    delta: number,
) {
    if (impulseFactor === 0) return;
    impulseVector.x = 0;
    impulseVector.y = -impulseFactor * delta / 1000;
    applyRotationToVector(impulseVector, impulseVector, vehicleRotation);
    Impulse.add(trackEid, impulseVector.x, impulseVector.y);
}

