import { DI } from '../../DI';
import { Changed, defineQuery } from 'bitecs';
import { Hitable } from '../Components/Hitable.ts';
import { TankPart } from '../Components/Tank.ts';
import { Bullet } from '../Components/Bullet.ts';
import { removeRigidEntity } from '../../Physical/createRigid.ts';
import { removeChild } from '../Components/Children.ts';
import { Parent } from '../Components/Parent.ts';

export function createHitableSystem({ world, physicalWorld } = DI) {
    const tankPartsQuery = defineQuery([TankPart, Parent, Changed(Hitable)]);
    const bulletsQuery = defineQuery([Bullet, Changed(Hitable)]);

    return () => {
        const tankPartsEids = tankPartsQuery(world);
        const bulletsIds = bulletsQuery(world);

        for (let i = 0; i < tankPartsEids.length; i++) {
            const tankPartIed = tankPartsEids[i];
            const damage = Hitable.damage[tankPartIed];
            const tankEid = Parent.id[tankPartIed];

            if (damage > 0) {
                const joint = physicalWorld.getImpulseJoint(TankPart.jointId[tankPartIed]);
                joint && physicalWorld.removeImpulseJoint(joint, true);
                removeChild(tankEid, tankPartIed);
            }
        }

        for (let i = 0; i < bulletsIds.length; i++) {
            const bulletsId = bulletsIds[i];
            const damage = Hitable.damage[bulletsId];

            if (damage > 0) {
                removeRigidEntity(bulletsId);
            }
        }
    };
}