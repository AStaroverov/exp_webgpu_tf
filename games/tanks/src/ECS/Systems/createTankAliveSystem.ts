import { DI } from '../../DI';
import { Changed, defineQuery, removeEntity } from 'bitecs';
import { Tank, TankPart } from '../Components/Tank.ts';
import { Children } from '../Components/Children.ts';

const MIN_ALIVE_PARTS = 190;

export function createTankAliveSystem({ world, physicalWorld } = DI) {
    const tanksQuery = defineQuery([Tank, Changed(Children)]);

    return () => {
        const tanksEids = tanksQuery(world);

        for (let i = 0; i < tanksEids.length; i++) {
            const tankEid = tanksEids[i];

            if (Children.entitiesCount[tankEid] < MIN_ALIVE_PARTS) {
                for (const tankPartEid of Children.entitiesIds[tankEid]) {
                    const joinId = TankPart.jointId[tankPartEid];
                    const joint = physicalWorld.getImpulseJoint(joinId);
                    joint && physicalWorld.removeImpulseJoint(joint, true);
                }
                removeEntity(world, tankEid);
            }
        }
    };
}