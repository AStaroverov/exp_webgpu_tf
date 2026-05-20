import { query } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { getGameComponents } from '../createGameWorld.ts';

const impulseVec = { x: 0, y: 0 };
const pointVec = { x: 0, y: 0 };

export function createApplyImpulseSystem({ world, physicalWorld } = GameDI) {
    const { RigidBodyRef, Impulse, TorqueImpulse, ImpulseAtPoint } = getGameComponents(world);

    return () => {
        const entitiesWithImpulse = query(world, [RigidBodyRef, Impulse]);

        for (let i = 0; i < entitiesWithImpulse.length; i++) {
            const eid = entitiesWithImpulse[i];

            if (!Impulse.hasImpulse(eid)) continue;

            const pid = RigidBodyRef.id[eid];
            if (pid === 0) continue;

            const rb = physicalWorld.getRigidBody(pid);
            if (rb == null) continue;

            impulseVec.x = Impulse.x[eid];
            impulseVec.y = Impulse.y[eid];

            rb.applyImpulse(impulseVec, true);
            Impulse.reset(eid);
        }

        const entitiesWithTorque = query(world, [RigidBodyRef, TorqueImpulse]);

        for (let i = 0; i < entitiesWithTorque.length; i++) {
            const eid = entitiesWithTorque[i];

            if (!TorqueImpulse.hasImpulse(eid)) continue;

            const pid = RigidBodyRef.id[eid];
            if (pid === 0) continue;

            const rb = physicalWorld.getRigidBody(pid);
            if (rb == null) continue;

            rb.applyTorqueImpulse(TorqueImpulse.value[eid], true);
            TorqueImpulse.reset(eid);
        }

        const entitiesWithPointImpulse = query(world, [RigidBodyRef, ImpulseAtPoint]);

        for (let i = 0; i < entitiesWithPointImpulse.length; i++) {
            const eid = entitiesWithPointImpulse[i];

            if (!ImpulseAtPoint.hasImpulse(eid)) continue;

            const pid = RigidBodyRef.id[eid];
            if (pid === 0) continue;

            const rb = physicalWorld.getRigidBody(pid);
            if (rb == null) continue;

            const count = ImpulseAtPoint.count[eid];
            for (let j = 0; j < count; j++) {
                const [ix, iy, px, py] = ImpulseAtPoint.get(eid, j);
                impulseVec.x = ix;
                impulseVec.y = iy;
                pointVec.x = px;
                pointVec.y = py;

                rb.applyImpulseAtPoint(impulseVec, pointVec, true);
            }

            ImpulseAtPoint.reset(eid);
        }
    };
}
