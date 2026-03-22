import { query } from "bitecs";
import { Vehicle } from "../../Game/ECS/Components/Vehicle";
import { MLState } from "./MlState";
import { GameDI } from "../../Game/DI/GameDI";

export function createMlScoreSystem({ world } = GameDI) {
    const tick = () => {
        if (!MLState.enabled) return;

        const vehicleEids = query(world, [Vehicle]);

        for (const vehicleEid of vehicleEids) {
            void vehicleEid;
            // Step rewards now come purely from event-based signals
            // (hitEnemy, killEnemy, gotHit, friendlyFire) tracked by other systems.
            // No more shaped rewards (exploration, proximity, adjacentEnemy).
        }
    };

    const dispose = () => {
    };

    return { tick, dispose };
}
