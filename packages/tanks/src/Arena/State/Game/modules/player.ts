import { distinctUntilChanged, interval, map } from "rxjs";
import { PlayerEnvDI } from "../../../../Game/DI/PlayerEnvDI";

export const playerVehicleEid$ = interval(100).pipe(
    map(() => PlayerEnvDI.tankEid),
    distinctUntilChanged(),
)