import { Scenario } from '../types.ts';
import { PilotType } from '../../../../Pilots/Components/Pilot.ts';

export function fillWithBots(
    episode: Scenario,
    type:
        | typeof PilotType.BotOnlyMoving
        | typeof PilotType.BotOnlyShooting
        | typeof PilotType.BotSimple
        | typeof PilotType.BotStrong,
) {
    const freeTanks = episode.getFreeTankEids();

    for (const tankEid of freeTanks) {
        episode.setPilot(tankEid, type);
    }
}

