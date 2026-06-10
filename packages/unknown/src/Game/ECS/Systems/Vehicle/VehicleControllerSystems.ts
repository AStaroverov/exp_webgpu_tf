import { GameDI } from '../../../DI/GameDI.ts';
import { hasComponent, query } from 'bitecs';
import { normalizeAngle } from '../../../../../../../lib/math.ts';
import { getGameComponents } from '../../createGameWorld.ts';

export function createVehicleTurretRotationSystem({ world } = GameDI) {
    const { VehicleTurret, TurretController, JointMotor, Parent, RigidBodyState, Slowed } = getGameComponents(world);

    return (delta: number) => {
        const turretEids = query(world, [VehicleTurret, TurretController, JointMotor]);

        for (let i = 0; i < turretEids.length; i++) {
            const turretEid = turretEids[i];
            const vehicleEid = Parent.id[turretEid];
            const vehicleRot = RigidBodyState.rotation[vehicleEid];
            const turretRot = RigidBodyState.rotation[turretEid];
            const turretRotDir = TurretController.rotation[turretEid];
            const maxRotationSpeed = VehicleTurret.rotationSpeed[turretEid];
            const slow = hasComponent(world, vehicleEid, Slowed) ? Slowed.slowMul[vehicleEid] : 1;

            const relTurretRot = normalizeAngle(turretRot - vehicleRot);
            const deltaRot = turretRotDir * maxRotationSpeed * slow * (delta / 1000);

            JointMotor.setTargetPosition$(turretEid, relTurretRot + deltaRot);
        }
    };
}
