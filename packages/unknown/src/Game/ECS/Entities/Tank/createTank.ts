import { createLightTank } from "./Light/LightTank.ts";
import { createMediumTank } from "./Medium/MediumTank.ts";
import { createStreamTank } from "./Medium/StreamTank.ts";
import { createEmpTank } from "./Medium/EmpTank.ts";
import { createHeavyTank } from "./Heavy/HeavyTank.ts";
import { createRocketTank } from "./Rocket/RocketTank.ts";
import { VehicleType } from "../../../Config/index.ts";
import { EntityId } from "bitecs";

export type TankVehicleType =
  | typeof VehicleType.LightTank
  | typeof VehicleType.MediumTank
  | typeof VehicleType.HeavyTank
  | typeof VehicleType.RocketTank
  | typeof VehicleType.FlameTank
  | typeof VehicleType.FrostTank
  | typeof VehicleType.EmpTank;

type TankOptions =
  | ({ type: typeof VehicleType.LightTank } & Parameters<typeof createLightTank>[0])
  | ({ type: typeof VehicleType.MediumTank } & Parameters<typeof createMediumTank>[0])
  | ({ type: typeof VehicleType.HeavyTank } & Parameters<typeof createHeavyTank>[0])
  | ({ type: typeof VehicleType.RocketTank } & Parameters<typeof createRocketTank>[0])
  | Parameters<typeof createStreamTank>[0]
  | ({ type: typeof VehicleType.EmpTank } & Parameters<typeof createEmpTank>[0]);

export function createTank(options: TankOptions): EntityId {
  const type = options.type;

  switch (type) {
    case VehicleType.LightTank:
      return createLightTank(options);
    case VehicleType.MediumTank:
      return createMediumTank(options);
    case VehicleType.HeavyTank:
      return createHeavyTank(options);
    case VehicleType.RocketTank:
      return createRocketTank(options);
    case VehicleType.FlameTank:
    case VehicleType.FrostTank:
      return createStreamTank(options);
    case VehicleType.EmpTank:
      return createEmpTank(options);
    default:
      throw new Error(`Unknown tank type ${type}`);
  }
}
