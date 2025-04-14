import { GameDI } from '../../DI/GameDI.ts';
import { Tank } from '../Components/Tank/Tank.ts';
import { Children } from '../Components/Children.ts';
import { CollisionGroup } from '../../Physical/createRigid.ts';
import { resetCollisionsTo } from '../../Physical/collision.ts';
import { removePhysicalJoint } from '../../Physical/joint.ts';
import { query } from 'bitecs';
import { TankPart } from '../Components/TankPart.ts';
import { getTankHealth } from '../Components/Tank/TankHealth.ts';
import { removeTankComponentsWithoutParts, resetTankPartJointComponent } from '../Components/Tank/TankUtils.ts';

export function createTankAliveSystem({ world } = GameDI) {
    return () => {
        const tankEids = query(world, [Tank, Children]);

        for (const tankEid of tankEids) {
            const hp = getTankHealth(tankEid);

            if (hp === 0) {
                // turret
                const turretEid = Tank.turretEId[tankEid];
                for (let i = 0; i < Children.entitiesCount[turretEid]; i++) {
                    breakPartFromTank(turretEid, i);
                }
                // tank parts
                for (let i = 0; i < Children.entitiesCount[tankEid]; i++) {
                    breakPartFromTank(tankEid, i);
                }
                removeTankComponentsWithoutParts(tankEid);
            }
        }
    };
}

function breakPartFromTank(eid: number, index: number) {
    const tankPartEid = Children.entitiesIds.get(eid, index);
    const jointPid = TankPart.jointPid[tankPartEid];

    if (jointPid === 0) return;

    // remove joints
    removePhysicalJoint(jointPid);
    resetTankPartJointComponent(tankPartEid);
    // change collision group
    resetCollisionsTo(tankPartEid, CollisionGroup.ALL & ~CollisionGroup.TANK_BASE);
}