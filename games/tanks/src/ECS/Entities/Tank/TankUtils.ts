import { Children } from '../../Components/Children.ts';
import { scheduleRemoveEntity } from '../../Utils/typicalRemoveEntity.ts';
import { TankPart, TankPartTrack } from '../../Components/TankPart.ts';
import { Tank } from '../../Components/Tank.ts';
import { GameDI } from '../../../DI/GameDI.ts';
import { Parent } from '../../Components/Parent.ts';
import { removePhysicalJoint } from '../../../Physical/joint.ts';
import { resetCollisionsTo } from '../../../Physical/collision.ts';
import { CollisionGroup } from '../../../Physical/createRigid.ts';
import { min, smoothstep } from '../../../../../../lib/math.ts';
import { PlayerRef } from '../../Components/PlayerRef.ts';
import { Score } from '../../Components/Score.ts';

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
} // return from 0 to 1
export const HEALTH_THRESHOLD = 0.75;

export function getTankHealth(tankEid: number): number {
    const initialPartsCount = Tank.initialPartsCount[tankEid];
    const partsCount = getTankCurrentPartsCount(tankEid);
    const absHealth = min(1, partsCount / initialPartsCount);
    const health = smoothstep(HEALTH_THRESHOLD, 1, absHealth);

    return health;
}

export function getTankScore(tankEid: number): number {
    const playerId = PlayerRef.id[tankEid];
    const score = Score.negativeScore[playerId] + Score.positiveScore[playerId];

    return score;
}