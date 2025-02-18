import { DI } from '../../DI';
import { Changed, defineQuery } from 'bitecs';
import { Hitable } from '../Components/Hitable.ts';
import { removeTankPartJointComponent, TankPart } from '../Components/Tank.ts';
import { Bullet } from '../Components/Bullet.ts';
import { CollisionGroup } from '../../Physical/createRigid.ts';
import { removeChild } from '../Components/Children.ts';
import { Parent } from '../Components/Parent.ts';
import { Wall } from '../Components/Wall.ts';
import { resetCollisionsTo } from '../../Physical/collision.ts';
import { removePhysicalJoint } from '../../Physical/joint.ts';
import { recursiveTypicalRemoveEntity } from '../Utils/typicalRemoveEntity.ts';

export function createHitableSystem({ world } = DI) {
    const tankPartsQuery = defineQuery([TankPart, Parent, Changed(Hitable)]);
    const bulletsQuery = defineQuery([Bullet, Changed(Hitable)]);
    const wallsQuery = defineQuery([Wall, Changed(Hitable)]);

    return () => {
        const tankPartsEids = tankPartsQuery(world);

        for (let i = 0; i < tankPartsEids.length; i++) {
            const tankPartEid = tankPartsEids[i];
            const damage = Hitable.damage[tankPartEid];
            const tankEid = Parent.id[tankPartEid];
            const jointPid = TankPart.jointPid[tankPartEid];

            if (damage > 0 && jointPid >= 0) {
                removeChild(tankEid, tankPartEid);
                resetCollisionsTo(tankPartEid, CollisionGroup.ALL & ~CollisionGroup.TANK_GUN);
                removePhysicalJoint(jointPid);
                removeTankPartJointComponent(tankPartEid);
            }
        }

        const bulletsIds = bulletsQuery(world);
        for (let i = 0; i < bulletsIds.length; i++) {
            const bulletsId = bulletsIds[i];
            const damage = Hitable.damage[bulletsId];

            if (damage > 0) {
                recursiveTypicalRemoveEntity(bulletsId);
            }
        }

        const wallsIds = wallsQuery(world);
        for (let i = 0; i < wallsIds.length; i++) {
            const wallId = wallsIds[i];
            const damage = Hitable.damage[wallId];

            if (damage > 0) {
                recursiveTypicalRemoveEntity(wallId);
            }
        }
    };
}