import { addComponent, EntityId, World } from 'bitecs';
import { delegate } from '../../../../../../src/delegate.ts';
import { TypedArray } from '../../../../../../src/utils.ts';
import { component } from '../../../../../../src/ECS/utils.ts';

export const Score = component({
    // Score show how many points the tank has
    negativeScore: TypedArray.f64(delegate.defaultSize),
    positiveScore: TypedArray.f64(delegate.defaultSize),

    addComponent(world: World, playerEid: EntityId): void {
        addComponent(world, playerEid, Score);
    },

    updateScore(playerEid: number, delta: number) {
        if (delta > 0) {
            Score.positiveScore[playerEid] += delta;
        } else {
            Score.negativeScore[playerEid] += delta;
        }
    },
});

