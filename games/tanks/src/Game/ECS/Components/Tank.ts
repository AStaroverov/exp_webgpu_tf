import { delegate } from '../../../../../../src/delegate.ts';
import { TypedArray } from '../../../../../../src/utils.ts';
import { component } from '../../../../../../src/ECS/utils.ts';
import { addComponent, EntityId, World } from 'bitecs';

export const Tank = component({
    turretEId: TypedArray.f64(delegate.defaultSize),
    initialPartsCount: TypedArray.f64(delegate.defaultSize),
    caterpillarsLength: TypedArray.f64(delegate.defaultSize),

    addComponent(world: World, eid: EntityId, initialPartsCount: number): void {
        Tank.turretEId[eid] = 0;
        Tank.caterpillarsLength[eid] = 0;

        addComponent(world, eid, Tank);
        Tank.initialPartsCount[eid] = initialPartsCount;
    },

    setTurretEid(eid: number, turretEid: number) {
        Tank.turretEId[eid] = turretEid;
    },

    setCaterpillarsLength(eid: number, caterpillarsLength: number) {
        Tank.caterpillarsLength[eid] = caterpillarsLength;
    },
});

