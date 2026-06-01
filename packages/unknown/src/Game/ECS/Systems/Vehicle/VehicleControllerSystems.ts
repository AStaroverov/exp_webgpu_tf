import { hasComponent, query } from 'bitecs';
import { normalizeAngle } from '../../../../../../../lib/math.ts';
import { getPhysicsWorldComponents } from '../../createPhysicsWorld.ts';
import { getBrainWorldComponents } from '../../createBrainWorld.ts';
import { getNodeParent, getNodePhysics } from '../../refs.ts';
import { Worlds } from '../../../DI/Worlds.ts';

export function createVehicleTurretRotationSystem({ physicsWorld, brainWorld } = Worlds) {
    const { VehicleTurret, JointMotor, RigidBodyState } = getPhysicsWorldComponents(physicsWorld);
    const { TurretController } = getBrainWorldComponents(brainWorld);

    return (delta: number) => {
        // Node-rooted: iterate turret NODES (every turret carries a TurretController
        // brain — set in createVehicleTurret), descending to the turret atom. Equals
        // the old query([VehicleTurret, JointMotor]) set, guarded by JointMotor below.
        const turretBrains = query(brainWorld, [TurretController]);

        for (let i = 0; i < turretBrains.length; i++) {
            const turretBrain = turretBrains[i];
            const turretEid = getNodePhysics(turretBrain);
            if (turretEid === 0 || !hasComponent(physicsWorld, turretEid, JointMotor)) continue;
            // turret node -> Brain parent (hull node) -> hull physics (vehicle atom).
            const vehicleEid = getNodePhysics(getNodeParent(turretBrain));
            const vehicleRot = RigidBodyState.rotation[vehicleEid];
            const turretRot = RigidBodyState.rotation[turretEid];
            const turretRotDir = TurretController.rotation[turretBrain];
            const maxRotationSpeed = VehicleTurret.rotationSpeed[turretEid];

            const relTurretRot = normalizeAngle(turretRot - vehicleRot);
            const deltaRot = turretRotDir * maxRotationSpeed * (delta / 1000);

            JointMotor.setTargetPosition$(turretEid, normalizeAngle(relTurretRot + deltaRot));
        }
    };
}
