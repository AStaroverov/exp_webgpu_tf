import { query } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { RigidBodyRef } from '../Components/Physical.ts';
import { Impulse, TorqueImpulse, ImpulseAtPoint } from '../Components/Impulse.ts';

const impulseVec = { x: 0, y: 0 };
const pointVec = { x: 0, y: 0 };

/**
 * System that applies accumulated impulses to rigid bodies.
 * Should run before physics simulation step.
 * After applying, impulses are reset to zero.
 */
export function createApplyImpulseSystem({ world, physicalWorld } = GameDI) {
    return () => {
        // Apply linear impulses at center of mass
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

        // Apply torque impulses
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

        // Apply impulses at specific world positions
        // This creates realistic torque when force is applied off-center
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
                
                // applyImpulseAtPoint applies force at specific world position
                // This creates both linear acceleration and angular acceleration
                // when the point is not at the center of mass
                rb.applyImpulseAtPoint(impulseVec, pointVec, true);
            }
            
            ImpulseAtPoint.reset(eid);
        }
    };
}

