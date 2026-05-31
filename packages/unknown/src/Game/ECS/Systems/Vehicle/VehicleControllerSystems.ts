import { query } from 'bitecs';
import { normalizeAngle } from '../../../../../../../lib/math.ts';
import { getPhysicsWorldComponents } from '../../createPhysicsWorld.ts';
import { getRenderWorldComponents } from '../../createRenderWorld.ts';
import { BridgeDI } from '../../../DI/BridgeDI.ts';
import { Worlds } from '../../../DI/Worlds.ts';

export function createVehicleTurretRotationSystem({ physicsWorld, renderWorld } = Worlds) {
    const { VehicleTurret, TurretController, JointMotor, RigidBodyState } = getPhysicsWorldComponents(physicsWorld);

    return (delta: number) => {
        const { Parent } = getRenderWorldComponents(renderWorld);
        const turretEids = query(physicsWorld, [VehicleTurret, TurretController, JointMotor]);

        for (let i = 0; i < turretEids.length; i++) {
            const turretEid = turretEids[i];
            const vehicleEid = BridgeDI.getPhysicsOf(Parent.id[BridgeDI.getRenderOf(turretEid)]);
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
