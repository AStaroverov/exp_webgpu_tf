import { Children } from '../../Components/Children.ts';
import { scheduleRemoveEntity } from '../../Utils/typicalRemoveEntity.ts';
import { TankPart, TankPartCaterpillar } from '../../Components/TankPart.ts';
import { Tank } from '../../Components/Tank.ts';
import { GameDI } from '../../../DI/GameDI.ts';
import { Parent } from '../../Components/Parent.ts';
import { removePhysicalJoint } from '../../../Physical/removePhysicalJoint.ts';
import { setPhysicalCollisionGroup } from '../../../Physical/setPhysicalCollisionGroup.ts';
import { CollisionGroup } from '../../../Physical/createRigid.ts';
import { min, smoothstep } from '../../../../../../../lib/math.ts';
import { PlayerRef } from '../../Components/PlayerRef.ts';
import { Score } from '../../Components/Score.ts';
import { EntityId, hasComponent, removeComponent } from 'bitecs';
import { TeamRef } from '../../Components/TeamRef.ts';
import { changePhysicalDensity } from '../../../Physical/changePhysicalDensity.ts';

export function destroyTank(tankEid: EntityId) {
    // turret
    const turretEid = Tank.turretEId[tankEid];
    for (let i = 0; i < Children.entitiesCount[turretEid]; i++) {
        const eid = Children.entitiesIds.get(turretEid, i);
        tearOffTankPart(eid, false);
    }
    Children.removeAllChildren(turretEid);
    // tank parts
    for (let i = 0; i < Children.entitiesCount[tankEid]; i++) {
        const partEid = Children.entitiesIds.get(tankEid, i);
        tearOffTankPart(partEid, false);
    }
    Children.removeAllChildren(tankEid);
    removeTankComponentsWithoutParts(tankEid);
}

export function removeTankComponentsWithoutParts(tankEid: number) {
    const turretEid = Tank.turretEId[tankEid];
    Children.removeChild(tankEid, turretEid);
    scheduleRemoveEntity(turretEid, false);
    scheduleRemoveEntity(tankEid, false);
}

export function tearOffTankPart(tankPartEid: number, shouldBreakConnection: boolean = true, { world } = GameDI) {
    removeComponent(world, tankPartEid, TeamRef);
    removeComponent(world, tankPartEid, PlayerRef);

    const parentEid = Parent.id[tankPartEid];
    if (shouldBreakConnection) {
        Children.removeChild(parentEid, tankPartEid);
    }

    const jointPid = hasComponent(world, tankPartEid, TankPart) ? TankPart.jointPid[tankPartEid] : 0;
    if (jointPid > 0) {
        removeComponent(world, tankPartEid, TankPart);
        resetTankPartJointComponent(tankPartEid);
        setPhysicalCollisionGroup(tankPartEid, CollisionGroup.ALL & ~CollisionGroup.TANK_BASE);
        removePhysicalJoint(jointPid);
        changePhysicalDensity(tankPartEid, 8);
    }
}

export function resetTankPartJointComponent(tankPartEid: number, { world } = GameDI) {
    TankPart.resetComponent(tankPartEid);
    TankPartCaterpillar.removeComponent(world, tankPartEid);
}

export function getTankCurrentPartsCount(tankEid: number) {
    return Children.entitiesCount[tankEid] + Children.entitiesCount[Tank.turretEId[tankEid]];
}

export const HEALTH_THRESHOLD = 0.85;

// return from 0 to 1
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

export function getTankTeamId(tankEid: number) {
    const teamId = TeamRef.id[tankEid];
    return teamId;
}
