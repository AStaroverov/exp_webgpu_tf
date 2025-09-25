import { addComponent, EntityId, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { component } from '../../../../../renderer/src/ECS/utils.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';

export const Score = component({
    negativeScore: TypedArray.f64(delegate.defaultSize),
    positiveScore: TypedArray.f64(delegate.defaultSize),

    addComponent(world: World, playerEid: EntityId): void {
        addComponent(world, playerEid, Score);
        Score.negativeScore[playerEid] = 0;
        Score.positiveScore[playerEid] = 0;
    },

    updateScore(playerEid: number, delta: number) {
        if (delta > 0) {
            Score.positiveScore[playerEid] += delta;
        } else {
            Score.negativeScore[playerEid] += delta;
        }
    },
});

