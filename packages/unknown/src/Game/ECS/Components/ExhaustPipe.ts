import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { addComponent, World } from 'bitecs';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';

export const createExhaustPipeComponent = defineComponent((ExhaustPipe) => {
    const relativeX = TypedArray.f32(delegate.defaultSize);
    const relativeY = TypedArray.f32(delegate.defaultSize);
    const direction = TypedArray.f32(delegate.defaultSize);
    const emissionRate = TypedArray.f32(delegate.defaultSize);
    const emissionAccumulator = TypedArray.f32(delegate.defaultSize);

    return {
        relativeX,
        relativeY,
        direction,
        emissionRate,
        emissionAccumulator,

        addComponent(world: World, eid: number, rx: number, ry: number, dir: number, rate: number) {
            addComponent(world, eid, ExhaustPipe);
            relativeX[eid] = rx;
            relativeY[eid] = ry;
            direction[eid] = dir;
            emissionRate[eid] = rate;
            emissionAccumulator[eid] = 0;
        },
    };
});
