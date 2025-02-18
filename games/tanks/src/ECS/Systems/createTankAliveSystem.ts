import { DI } from '../../DI';
import { Changed, defineQuery } from 'bitecs';
import {
    getTankHealth,
    removeTankComponentsWithoutParts,
    resetTankPartJointComponent,
    Tank,
    TankPart,
} from '../Components/Tank.ts';
import { Children } from '../Components/Children.ts';
import { CollisionGroup } from '../../Physical/createRigid.ts';
import { resetCollisionsTo } from '../../Physical/collision.ts';
import { removePhysicalJoint } from '../../Physical/joint.ts';

export function createTankAliveSystem({ world } = DI) {
    const tanksQuery = defineQuery([Tank, Changed(Children)]);

    return () => {
        const tankEids = tanksQuery(world);

        for (const tankEid of tankEids) {
            const hp = getTankHealth(tankEid);

            if (hp < 0.9) {
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
    const tankPartEid = Children.entitiesIds[eid][index];
    const jointPid = TankPart.jointPid[tankPartEid];
    // remove joints
    removePhysicalJoint(jointPid);
    resetTankPartJointComponent(tankPartEid);
    // change collision group
    resetCollisionsTo(tankPartEid, CollisionGroup.ALL & ~CollisionGroup.TANK_BASE);
}