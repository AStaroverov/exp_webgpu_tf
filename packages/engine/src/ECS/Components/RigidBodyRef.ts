import { addComponent, World } from "bitecs";
import { delegate } from "../../../../renderer3d_2/src/delegate.ts";
import { defineComponent } from "../../../../renderer3d_2/src/ECS/utils.ts";

// Bridge component: stores the Rapier body handle (pid) per entity, and maintains
// the reverse pid→eid map for collision-event resolution. Handle 0 is the
// reserved empty-memory sentinel (see initPhysicalWorld.reserveHandleZero).
const mapPhysicalIdToEntityId = new Map<number, number>();

export const createRigidBodyRefComponent = defineComponent((RigidBodyRef) => {
  const id = new Float64Array(delegate.defaultSize);
  return {
    id,
    addComponent(world: World, eid: number, pid: number) {
      addComponent(world, eid, RigidBodyRef);
      id[eid] = pid;
      mapPhysicalIdToEntityId.set(pid, eid);
    },
    clear(eid: number) {
      const pid = id[eid];
      if (pid !== 0) {
        id[eid] = 0;
        mapPhysicalIdToEntityId.delete(pid);
      }
    },
    dispose() {
      mapPhysicalIdToEntityId.clear();
    },
  };
});

export function getEntityIdByPhysicalId(physicalId: number): number {
  const eid = mapPhysicalIdToEntityId.get(physicalId);
  if (eid === undefined) {
    throw new Error(`Entity with physicalId ${physicalId} not found`);
  }
  return eid;
}
