import { GameDI } from '../../DI/GameDI.ts';
import { Hitable } from '../Components/Hitable.ts';
import { CollisionGroup } from '../../Physical/createRigid.ts';
import { Children } from '../Components/Children.ts';
import { Parent } from '../Components/Parent.ts';
import { resetCollisionsTo } from '../../Physical/collision.ts';
import { removePhysicalJoint } from '../../Physical/joint.ts';
import { scheduleRemoveEntity } from '../Utils/typicalRemoveEntity.ts';
import { onSet, query } from 'bitecs';
import { Bullet } from '../Components/Bullet.ts';
import { createChangeDetector } from '../../../../../src/ECS/Systems/ChangedDetectorSystem.ts';
import { TankPart } from '../Components/TankPart.ts';
import { resetTankPartJointComponent } from '../Components/Tank/TankUtils.ts';

export function createHitableSystem({ world } = GameDI) {
    const hitableChanges = createChangeDetector(world, [onSet(Hitable)]);

    return () => {
        const tankPartsEids = query(world, [TankPart, Parent, Hitable]);

        for (let i = 0; i < tankPartsEids.length; i++) {
            const tankPartEid = tankPartsEids[i];
            if (!hitableChanges.has(tankPartEid)) continue;
            const damage = Hitable.damage[tankPartEid];
            const parentEid = Parent.id[tankPartEid];
            const jointPid = TankPart.jointPid[tankPartEid];

            if (damage > 0 && jointPid >= 0) {
                Children.removeChild(parentEid, tankPartEid);
                removePhysicalJoint(jointPid);
                resetCollisionsTo(tankPartEid, CollisionGroup.ALL & ~CollisionGroup.TANK_BASE);
                resetTankPartJointComponent(tankPartEid);
            }
        }

        const bulletsIds = query(world, [Bullet, Hitable]);
        for (let i = 0; i < bulletsIds.length; i++) {
            const bulletsId = bulletsIds[i];
            const damage = Hitable.damage[bulletsId];

            if (hitableChanges.has(bulletsId) && damage > 0) {
                scheduleRemoveEntity(bulletsId);
            }
        }

        // const wallsIds = query(world, [Wall, Hitable]);
        // for (let i = 0; i < wallsIds.length; i++) {
        //     const wallId = wallsIds[i];
        //     const damage = Hitable.damage[wallId];
        //
        //     if (hitableChanges.has(wallId) && damage > 0) {
        //         scheduleRemoveEntity(wallId);
        //     }
        // }

        hitableChanges.clear();
    };
}