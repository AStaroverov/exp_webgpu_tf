import { DI } from '../../DI';
import { Changed, defineQuery } from 'bitecs';
import { removeTankComponentsWithoutParts, removeTankPartJointComponent, Tank, TankPart } from '../Components/Tank.ts';
import { Children } from '../Components/Children.ts';
import { CollisionGroup } from '../../Physical/createRigid.ts';
import { resetCollisionsTo } from '../../Physical/collision.ts';
import { removePhysicalJoint } from '../../Physical/joint.ts';

const MIN_ALIVE_PARTS = 190;

export function createTankAliveSystem({ world } = DI) {
    const tanksQuery = defineQuery([Tank, Changed(Children)]);

    return () => {
        const tankEids = tanksQuery(world);

        for (const tankEid of tankEids) {
            const childrenLength = Children.entitiesCount[tankEid];

            if (childrenLength < MIN_ALIVE_PARTS) {
                for (let i = 0; i < childrenLength; i++) {
                    const tankPartEid = Children.entitiesIds[tankEid][i];
                    const jointPid = TankPart.jointPid[tankPartEid];
                    // remove joints
                    removePhysicalJoint(jointPid);
                    removeTankPartJointComponent(tankPartEid);
                    // change collision group
                    resetCollisionsTo(tankPartEid, CollisionGroup.ALL & ~CollisionGroup.TANK_GUN);
                }
                removeTankComponentsWithoutParts(tankEid);
            }
        }
    };
}