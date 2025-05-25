import { hasComponent, query } from 'bitecs';
import { GameDI } from '../../../DI/GameDI.ts';
import { Tank } from '../../Components/Tank.ts';
import { Children } from '../../Components/Children.ts';
import { TankPart, TankPartCaterpillar } from '../../Components/TankPart.ts';
import { RigidBodyState } from '../../Components/Physical.ts';
import { Vector2 } from '@dimforge/rapier2d-simd';
import { abs, cos, PI, sign, sin } from '../../../../../../../lib/math.ts';

export function createVisualizationTracksSystem({ world, physicalWorld } = GameDI) {
    const vect2 = new Vector2(0, 0);

    return (_delta: number) => {
        const tankEids = query(world, [Tank]);

        for (const tankEid of tankEids) {
            const caterpillarsLimit = Tank.caterpillarsLength[tankEid] / 2;
            const childrenEids = Children.entitiesIds.getBatch(tankEid);
            const childrenCount = Children.entitiesCount[tankEid];
            const linvel = RigidBodyState.linvel.getBatch(tankEid);
            const angvel = RigidBodyState.angvel[tankEid];
            const rotation = RigidBodyState.rotation[tankEid];

            const forwardX = cos(rotation - PI / 2);
            const forwardY = sin(rotation - PI / 2);
            const speed = linvel[0] * forwardX + linvel[1] * forwardY;

            for (let i = 0; i < childrenCount; i++) {
                const childEid = childrenEids[i];

                if (!hasComponent(world, childEid, TankPartCaterpillar)) continue;

                const jointPid = TankPart.jointPid[childEid];
                const joint = physicalWorld.getImpulseJoint(jointPid);

                if (jointPid === 0 || joint == null) continue;

                const anchor1 = TankPart.anchor1.getBatch(childEid);
                const angFactor = anchor1[0] > 0 ? -0.8 : 0.8;
                let delta = (speed / 100 + (angvel * angFactor));
                delta -= delta % 0.01;

                if (abs(delta) < 0.05) continue;

                anchor1[1] -= delta;
                anchor1[1] -= anchor1[1] % 0.01;

                if (abs(anchor1[1]) > caterpillarsLimit) {
                    anchor1[1] = -sign(anchor1[1]) * (caterpillarsLimit + (caterpillarsLimit - abs(anchor1[1])));
                }

                vect2.x = anchor1[0];
                vect2.y = anchor1[1];
                joint.setAnchor1(vect2);
            }
        }
    };
}