import { DI } from '../../DI';
import { Changed, defineQuery, removeEntity } from 'bitecs';
import { Tank, TankPart } from '../Components/Tank.ts';
import { Children } from '../Components/Children.ts';

const MIN_ALIVE_PARTS = 190;

export function createTankAliveSystem({ world, physicalWorld } = DI) {
    const tanksQuery = defineQuery([Tank, Changed(Children)]);

    return () => {
        const tanksEids = tanksQuery(world);

        for (const tanksEid of tanksEids) {
            if (Children.entitiesCount[tanksEid] < MIN_ALIVE_PARTS) {
                for (const tankPartEid of Children.entitiesIds[tanksEid]) {
                    const joinId = TankPart.jointId[tankPartEid];
                    const joint = physicalWorld.getImpulseJoint(joinId);
                    joint && physicalWorld.removeImpulseJoint(joint, true);
                }
                removeEntity(world, tanksEid);
            }
        }
    };
}