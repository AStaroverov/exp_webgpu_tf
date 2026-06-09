import { GameDI } from '../../../DI/GameDI.ts';
import { query } from 'bitecs';
import { normalizeAngle } from '../../../../../../../lib/math.ts';
import { getGameComponents } from '../../createGameWorld.ts';

export function createVehicleTurretRotationSystem({ world } = GameDI) {
    const { VehicleTurret, TurretController, JointMotor, Parent, RigidBodyState } = getGameComponents(world);

    return (delta: number) => {
        const turretEids = query(world, [VehicleTurret, TurretController, JointMotor]);

        for (let i = 0; i < turretEids.length; i++) {
            const turretEid = turretEids[i];
            const vehicleEid = Parent.id[turretEid];
            const vehicleRot = RigidBodyState.rotation[vehicleEid];
            const turretRot = RigidBodyState.rotation[turretEid];
            const turretRotDir = TurretController.rotation[turretEid];
            const maxRotationSpeed = VehicleTurret.rotationSpeed[turretEid];

            const relTurretRot = normalizeAngle(turretRot - vehicleRot);
            const deltaRot = turretRotDir * maxRotationSpeed * (delta / 1000);

            JointMotor.setTargetPosition$(turretEid, relTurretRot + deltaRot);
        }
    };
}
