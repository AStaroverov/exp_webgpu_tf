import { TypedArray } from '../../../../../../../renderer/src/utils.ts';
import { delegate } from '../../../../../../../renderer/src/delegate.ts';
import { addComponent, World } from 'bitecs';
import { defineComponent } from '../../../../../../../renderer/src/ECS/utils.ts';

/** Parameters for a TurretAim action. `tolerance` = aim accuracy (radians). */
export const createTurretAimParamsComponent = defineComponent((TurretAimParams) => {
    const tolerance = TypedArray.f64(delegate.defaultSize);
    return {
        tolerance,
        addComponent(world: World, eid: number, t: number = 0.05) {
            addComponent(world, eid, TurretAimParams);
            tolerance[eid] = t;
        },
    };
});
