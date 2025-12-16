import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { component } from '../../../../../renderer/src/ECS/utils.ts';
import { addComponent, EntityId, World } from 'bitecs';
import { TankEngineType } from '../Systems/Tank/TankControllerSystems.ts';

export enum TankType {
    Light = 0,
    Medium = 1,
    Heavy = 2,
    Player = 3,  // Special player tank - medium size but faster
}

export const Tank = component({
    turretEId: TypedArray.f64(delegate.defaultSize),

    type: TypedArray.i8(delegate.defaultSize),
    engineType: TypedArray.i8(delegate.defaultSize),
    initialPartsCount: TypedArray.f64(delegate.defaultSize),
    caterpillarsLength: TypedArray.f64(delegate.defaultSize),

    addComponent(world: World, eid: EntityId, type: TankType, initialPartsCount: number): void {
        Tank.turretEId[eid] = 0;
        Tank.engineType[eid] = 0;
        Tank.caterpillarsLength[eid] = 0;

        addComponent(world, eid, Tank);
        Tank.type[eid] = type;
        Tank.initialPartsCount[eid] = initialPartsCount;
    },

    setTurretEid(eid: number, turretEid: number) {
        Tank.turretEId[eid] = turretEid;
    },

    setEngineType(eid: number, engineType: TankEngineType) {
        Tank.engineType[eid] = engineType;
    },

    setCaterpillarsLength(eid: number, caterpillarsLength: number) {
        Tank.caterpillarsLength[eid] = caterpillarsLength;
    },
});

