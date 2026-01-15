import { randomRangeFloat } from '../../../../lib/random.ts';
import { SimpleBotFeatures } from '../../../tanks/src/Pilots/Agents/SimpleBot.ts';

export type BotLevel = 0 | 1 | 2;

export function createBotFeatures(level: BotLevel): SimpleBotFeatures {
    switch (level) {
        case 0:
            return {
                move: randomRangeFloat(0.1, 0.5),
                aim: {
                    aimError: randomRangeFloat(0.5, 0.8),
                    shootChance: randomRangeFloat(0.01, 0.3),
                },
            };
        case 1:
            return {
                move: randomRangeFloat(0.5, 1),
                aim: {
                    aimError: randomRangeFloat(0.3, 0.5),
                    shootChance: randomRangeFloat(0.3, 0.8),
                },
            };
        case 2:
            return {
                move: randomRangeFloat(0.8, 1),
                aim: {
                    aimError: randomRangeFloat(0.1, 0.3),
                    shootChance: randomRangeFloat(0.5, 0.9),
                },
            };
    }
}

