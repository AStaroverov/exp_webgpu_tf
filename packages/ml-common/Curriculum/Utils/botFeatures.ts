import { randomRangeFloat } from '../../../../lib/random.ts';
import { SimpleBotFeatures } from '../../../tanks/src/Plugins/Pilots/Agents/SimpleBot.ts';

export type BotLevel = 0 | 1 | 2;

export function createBotFeatures(level: BotLevel): SimpleBotFeatures {
    switch (level) {
        case 0:
            return {
                move: randomRangeFloat(0.1, 0.2),
                aim: {
                    aimError: randomRangeFloat(0.5, 0.8),
                    shootChance: randomRangeFloat(0.01, 0.2),
                },
            };
        case 1:
            return {
                move: randomRangeFloat(0.2, 0.6),
                aim: {
                    aimError: randomRangeFloat(0.3, 0.5),
                    shootChance: randomRangeFloat(0.3, 0.8),
                },
            };
        case 2:
            return {
                move: randomRangeFloat(0.6, 1),
                aim: {
                    aimError: randomRangeFloat(0.1, 0.3),
                    shootChance: randomRangeFloat(0.5, 0.9),
                },
            };
    }
}

