import { GameDI } from '../../Game/DI/GameDI.ts';
import { Pilot, PilotAgents, PilotType } from '../Components/Pilot.ts';
import { EntityId, query } from 'bitecs';
import { CurrentActorAgent } from '../Agents/CurrentActorAgent.ts';
import { VersionedAgent } from '../Agents/VersionedAgent.ts';
import { ValueOf } from '../../../../../lib/Types';
import { RandomHistoricalAgent } from '../Agents/RandomHistoricalAgent.ts';
import { SimpleBot } from '../Agents/SimpleBot.ts';
import { randomRangeFloat } from '../../../../../lib/random.ts';

export function createInitPilotSystem({ world, setPlayerTank } = GameDI) {
    return () => {
        const pilotEids = query(world, [Pilot]);

        if (PilotAgents.size === pilotEids.length) return;

        for (let i = 0; i < pilotEids.length; i++) {
            const pilotEid = pilotEids[i];
            const type = Pilot.type[pilotEid];

            if (type === PilotType.Player) {
                setPlayerTank(pilotEid);
            }

            if (!PilotAgents.has(pilotEid)) {
                const agent = createAgent(pilotEid, type);
                PilotAgents.set(pilotEid, agent);
            }
        }
    };
}

function createAgent(tankEid: EntityId, version: ValueOf<typeof PilotType>) {
    switch (version) {
        case PilotType.Player:
            return undefined;
        case PilotType.AgentLearnable:
            return new CurrentActorAgent(tankEid, true);
        case PilotType.Agent:
            return new CurrentActorAgent(tankEid, false);
        case PilotType.AgentRandom:
            return new RandomHistoricalAgent(tankEid);
        case PilotType.Agent31:
        case PilotType.Agent32:
            return new VersionedAgent(tankEid, `assets/models/v${ version }`);
        case PilotType.BotOnlyMoving:
            return new SimpleBot(tankEid, {
                move: randomRangeFloat(0.4, 1),
            });
        case PilotType.BotOnlyShooting:
            return new SimpleBot(tankEid, {
                aim: {
                    aimError: randomRangeFloat(0.05, 0.1),
                    shootChance: randomRangeFloat(0.2, 0.4),
                },
            });
        case PilotType.BotSimple:
            return new SimpleBot(tankEid, {
                move: randomRangeFloat(0.4, 0.8),
                aim: {
                    aimError: randomRangeFloat(0.05, 0.1),
                    shootChance: randomRangeFloat(0.2, 0.4),
                },
            });
        case PilotType.BotStrong:
            return new SimpleBot(tankEid, {
                move: 1,
                aim: {
                    aimError: 0,
                    shootChance: 0.8,
                },
            });
        default:
            throw new Error(`Unknown version ${ version }`);
    }
}
