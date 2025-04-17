import { Children } from '../Children.ts';
import { scheduleRemoveEntity } from '../../Utils/typicalRemoveEntity.ts';
import { TankPart, TankPartTrack } from './TankPart.ts';
import { Tank } from './Tank.ts';
import { GameDI } from '../../../DI/GameDI.ts';
import { Parent } from '../Parent.ts';
import { removePhysicalJoint } from '../../../Physical/joint.ts';
import { resetCollisionsTo } from '../../../Physical/collision.ts';
import { CollisionGroup } from '../../../Physical/createRigid.ts';

export function removeTankComponentsWithoutParts(tankEid: number) {
    const aimEid = Tank.aimEid[tankEid];
    const turretEid = Tank.turretEId[tankEid];
    Children.removeChild(tankEid, aimEid);
    Children.removeChild(tankEid, turretEid);
    scheduleRemoveEntity(aimEid, false);
    scheduleRemoveEntity(turretEid, false);
    scheduleRemoveEntity(tankEid, false);
}

export function tearOffTankPart(tankPartEid: number) {
    const parentEid = Parent.id[tankPartEid];
    const jointPid = TankPart.jointPid[tankPartEid];

    if (jointPid >= 0) {
        Children.removeChild(parentEid, tankPartEid);
        removePhysicalJoint(jointPid);
        resetCollisionsTo(tankPartEid, CollisionGroup.ALL & ~CollisionGroup.TANK_BASE);
        resetTankPartJointComponent(tankPartEid);
    }
}

export function resetTankPartJointComponent(tankPartEid: number, { world } = GameDI) {
    TankPart.resetComponent(tankPartEid);
    TankPartTrack.removeComponent(world, tankPartEid);
}

export function getTankCurrentPartsCount(tankEid: number) {
    return Children.entitiesCount[tankEid] + Children.entitiesCount[Tank.turretEId[tankEid]];
}