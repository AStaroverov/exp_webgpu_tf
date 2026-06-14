import { GameDI } from "../../../DI/GameDI.ts";
import { hasComponent, query } from "bitecs";
import { normalizeAngle } from "../../../../../../../lib/math.ts";
import { getGameComponents } from "../../createGameWorld.ts";

export function createVehicleTurretRotationSystem({ world } = GameDI) {
  const { VehicleTurret, TurretController, JointMotor, Parent, RigidBodyState, Slowed, Stunned } =
    getGameComponents(world);

  return (delta: number) => {
    const turretEids = query(world, [VehicleTurret, TurretController, JointMotor]);

    for (let i = 0; i < turretEids.length; i++) {
      const turretEid = turretEids[i];
      const vehicleEid = Parent.id.get(turretEid);
      const vehicleRot = RigidBodyState.rotation[vehicleEid];
      const turretRot = RigidBodyState.rotation[turretEid];
      const turretRotDir = TurretController.rotation.get(turretEid);
      const maxRotationSpeed = VehicleTurret.rotationSpeed.get(turretEid);
      const slow = 1 - Slowed.slowMul.get(vehicleEid);
      const stun = hasComponent(world, vehicleEid, Stunned) ? 0 : 1;

      const deltaRot = turretRotDir * maxRotationSpeed * slow * stun * (delta / 1000);
      if (deltaRot === 0) continue;

      const relTurretRot = normalizeAngle(turretRot - vehicleRot);
      JointMotor.setTargetPosition$(turretEid, relTurretRot + deltaRot);
    }
  };
}
