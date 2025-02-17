import { DI } from '../../DI';
import { Changed, defineQuery } from 'bitecs';
import { removeTankPartJoint, removeTankWithoutParts, Tank, TankPart } from '../Components/Tank.ts';
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
            if (Children.entitiesCount[tankEid] < MIN_ALIVE_PARTS) {
                for (let i = 0; i < Children.entitiesCount[tankEid]; i++) {
                    const tankPartEid = Children.entitiesIds[tankEid][i];
                    // remove joints
                    removePhysicalJoint(TankPart.jointPid[tankPartEid]);
                    removeTankPartJoint(tankPartEid);
                    // change collision group
                    resetCollisionsTo(tankPartEid, CollisionGroup.WALL);
                }
                removeTankWithoutParts(tankEid);
            }
        }
    };
}