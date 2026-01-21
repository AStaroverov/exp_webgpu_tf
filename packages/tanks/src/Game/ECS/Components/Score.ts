import { addComponent, EntityId, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { component } from '../../../../../renderer/src/ECS/utils.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';

const POSITIVE_METRICS = [
    'hitEnemy',
    'killEnemy',
    'adjacentEnemyDetection',
    'exploration',
    'spices',
    'debris',
] as const;
const NEGATIVE_METRICS = [
    'friendlyFire',
    'gotHit',
    'proximityPenalty',
] as const;
const METRICS = [...POSITIVE_METRICS, ...NEGATIVE_METRICS] as const;
const ScoreData = METRICS.reduce(
    (acc, metric) => ({ ...acc, ...addStore(metric) }),
    {} as Record<(typeof METRICS)[number], Float64Array>
);
const ScoreMethods = METRICS.reduce(
    (acc, metric) => ({ ...acc, ...addMethod(metric) }),
    {} as Record<`add${Capitalize<(typeof METRICS)[number]>}`, (playerEid: number, amount: number) => void>
);
export const Score = component({
    ...ScoreData,
    ...ScoreMethods,

    addComponent(world: World, playerEid: EntityId): void {
        addComponent(world, playerEid, Score);
        METRICS.forEach(metric => Score[metric][playerEid] = 0);
    },

    getAllScores(playerEid: number): Record<(typeof METRICS)[number], number> {
        return METRICS.reduce((acc, metric) => ({ ...acc, [metric]: ScoreData[metric][playerEid] }), {} as Record<(typeof METRICS)[number], number>);
    },

    getPositiveScore(playerEid: number): number {
        return POSITIVE_METRICS.reduce((acc, metric) => acc + ScoreData[metric][playerEid], 0);
    },

    getNegativeScore(playerEid: number): number {
        return NEGATIVE_METRICS.reduce((acc, metric) => acc + ScoreData[metric][playerEid], 0);
    },

    getTotalScore(playerEid: number): number {
        return Score.getPositiveScore(playerEid) + Score.getNegativeScore(playerEid);
    },
});

function capitalize(str: string) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function addStore(metric: (typeof METRICS)[number]) {
    return { [metric]: TypedArray.f64(delegate.defaultSize) } as const;
}

function addMethod(metric: (typeof METRICS)[number]) {
    return {
        ['add' + capitalize(metric)]: (playerEid: number, amount: number) => { ScoreData[metric][playerEid] += amount; }
    } as const;
}
