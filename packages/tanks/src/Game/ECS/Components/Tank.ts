import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { component } from '../../../../../renderer/src/ECS/utils.ts';
import { addComponent, EntityId, World } from 'bitecs';

export const Tank = component({
    turretEId: TypedArray.f64(delegate.defaultSize),

    addComponent(world: World, eid: EntityId): void {
        addComponent(world, eid, Tank);
        Tank.turretEId[eid] = 0;
    },

    setTurretEid(eid: number, turretEid: number) {
        Tank.turretEId[eid] = turretEid;
    },
});

