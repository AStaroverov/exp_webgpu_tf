import { getPhysicsWorldComponents, PhysicsWorld } from '../ECS/createPhysicsWorld.ts';
import { getRenderWorldComponents, RenderGameWorld } from '../ECS/createRenderWorld.ts';

type LinkKind = 'mirror'; // Step 1 only owns physics<->render. (owner/homeSlot/occupant/carrier arrive Steps 2-3.)

export const BridgeDI = {
    physicsWorld: null as unknown as PhysicsWorld,
    renderWorld: null as unknown as RenderGameWorld,

    // physical body id (pid) -> physics atom eid (replaces mapPhysicalIdToEntityId)
    physicalIdToPhysics: new Map<number, number>(),

    init(physicsWorld: PhysicsWorld, renderWorld: RenderGameWorld) {
        this.physicsWorld = physicsWorld;
        this.renderWorld = renderWorld;
        this.physicalIdToPhysics.clear();
    },

    // ---- mutators ----
    link(kind: LinkKind, physEid: number, renderEid: number) {
        if (kind !== 'mirror') throw new Error('Bridge: unknown link kind ' + kind);
        const P = getPhysicsWorldComponents(this.physicsWorld);
        const R = getRenderWorldComponents(this.renderWorld);
        P.RenderRef.set(this.physicsWorld, physEid, renderEid);
        R.PhysicsRef.set(this.renderWorld, renderEid, physEid);
    },
    unlink(kind: LinkKind, physEid: number) {
        if (kind !== 'mirror') throw new Error('Bridge: unknown link kind ' + kind);
        const P = getPhysicsWorldComponents(this.physicsWorld);
        const R = getRenderWorldComponents(this.renderWorld);
        const renderEid = P.RenderRef.id[physEid];
        if (renderEid !== 0) R.PhysicsRef.clear(this.renderWorld, renderEid);
        P.RenderRef.clear(this.physicsWorld, physEid);
    },
    registerPhysicalId(pid: number, physEid: number) {
        this.physicalIdToPhysics.set(pid, physEid);
    },
    unregisterPhysicalId(pid: number) {
        this.physicalIdToPhysics.delete(pid);
    },

    // ---- translators (pure lookups) ----
    getRenderOf(physEid: number): number {
        return getPhysicsWorldComponents(this.physicsWorld).RenderRef.id[physEid];
    },
    getPhysicsOf(renderEid: number): number {
        return getRenderWorldComponents(this.renderWorld).PhysicsRef.id[renderEid];
    },
    getPhysicsByPhysicalId(pid: number): number | undefined {
        return this.physicalIdToPhysics.get(pid);
    },

    // ---- debug ----
    validate() {
        const P = getPhysicsWorldComponents(this.physicsWorld);
        const R = getRenderWorldComponents(this.renderWorld);
        // physics ⊆ render, bidirectional consistency.
        for (const [, physEid] of this.physicalIdToPhysics) {
            const renderEid = P.RenderRef.id[physEid];
            if (renderEid === 0) throw new Error(`Bridge.validate: physics atom ${physEid} has no mirror`);
            if (R.PhysicsRef.id[renderEid] !== physEid) {
                throw new Error(`Bridge.validate: mirror inconsistency for physics ${physEid} / render ${renderEid}`);
            }
        }
    },

    dispose() {
        this.physicalIdToPhysics.clear();
        this.physicsWorld = null!;
        this.renderWorld = null!;
    },
};
