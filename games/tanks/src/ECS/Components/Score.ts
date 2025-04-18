import { addComponent, EntityId, World } from 'bitecs';
import { delegate } from '../../../../../src/delegate';
import { TypedArray } from '../../../../../src/utils.ts';
import { component } from '../../../../../src/ECS/utils';

export const Score = component({
    // Score show how many points the tank has
    negativeScore: TypedArray.f64(delegate.defaultSize),
    positiveScore: TypedArray.f64(delegate.defaultSize),

    addComponent(world: World, eid: EntityId): void {
        addComponent(world, eid, Score);
    },

    updateScore(tankEid: number, delta: number) {
        if (delta > 0) {
            Score.positiveScore[tankEid] += delta;
        } else {
            Score.negativeScore[tankEid] += delta;
        }
    },
});

