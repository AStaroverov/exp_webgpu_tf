import { query } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { RigidBodyRef } from '../Components/Physical.ts';
import { Impulse, TorqueImpulse } from '../Components/Impulse.ts';

const impulseVec = { x: 0, y: 0 };

/**
 * System that applies accumulated impulses to rigid bodies.
 * Should run before physics simulation step.
 * After applying, impulses are reset to zero.
 */
export function createApplyImpulseSystem({ world, physicalWorld } = GameDI) {
    return () => {
        // Apply linear impulses
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
    };
}

