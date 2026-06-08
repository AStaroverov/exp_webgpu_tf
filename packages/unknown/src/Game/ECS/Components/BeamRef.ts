import { addComponent, EntityId, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';

export const createBeamRefComponent = defineComponent((BeamRef) => {
    const beamEid = TypedArray.f64(delegate.defaultSize);

    return {
        beamEid,

        addComponent(world: World, ownerEid: EntityId, beam: EntityId) {
            addComponent(world, ownerEid, BeamRef);
            beamEid[ownerEid] = beam;
        },

        getBeamEid: (ownerEid: EntityId) => beamEid[ownerEid],
    };
});
