import { GameDI } from '../../../DI/GameDI.ts';
import { TurretController } from '../../Components/TurretController.ts';
import { RigidBodyState } from '../../Components/Physical.ts';
import { query } from 'bitecs';
import { normalizeAngle } from '../../../../../../../lib/math.ts';
import { VehicleTurret } from '../../Components/VehicleTurret.ts';
import { Parent } from '../../Components/Parent.ts';
import { JointMotor } from '../../Components/JointMotor.ts';

export function createVehicleTurretRotationSystem({ world } = GameDI) {
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

            JointMotor.setTargetPosition$(turretEid, normalizeAngle(relTurretRot + deltaRot));
        }
    };
}
