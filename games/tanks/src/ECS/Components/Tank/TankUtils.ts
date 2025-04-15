import { Children } from '../Children.ts';
import { scheduleRemoveEntity } from '../../Utils/typicalRemoveEntity.ts';
import { TankPart, TankPartTrack } from './TankPart.ts';
import { Tank } from './Tank.ts';
import { GameDI } from '../../../DI/GameDI.ts';

export function removeTankComponentsWithoutParts(tankEid: number) {
    const aimEid = Tank.aimEid[tankEid];
    const turretEid = Tank.turretEId[tankEid];
    Children.removeChild(tankEid, aimEid);
    Children.removeChild(tankEid, turretEid);
    scheduleRemoveEntity(aimEid, false);
    scheduleRemoveEntity(turretEid, false);
    scheduleRemoveEntity(tankEid, false);
}

export function resetTankPartJointComponent(tankPartEid: number, { world } = GameDI) {
    TankPart.resetComponent(tankPartEid);
    TankPartTrack.removeComponent(world, tankPartEid);
}

export function getTankCurrentPartsCount(tankEid: number) {
    return Children.entitiesCount[tankEid] + Children.entitiesCount[Tank.turretEId[tankEid]];
}