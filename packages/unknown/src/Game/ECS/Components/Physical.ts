import { addComponent, EntityId, World } from "bitecs";
import { delegate } from "../../../../../renderer/src/delegate.ts";
import { NestedArray, TypedArray } from "../../../../../renderer/src/utils.ts";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

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
  if (!mapPhysicalIdToEntityId.has(physicalId))
    throw new Error(`Entity with physicalId ${physicalId} not found`);
  return mapPhysicalIdToEntityId.get(physicalId)!;
}

export const createRigidBodyStateComponent = defineComponent((RigidBodyState) => {
  const position = NestedArray.f64(2, delegate.defaultSize);
  const rotation = TypedArray.f64(delegate.defaultSize);
  const linvel = NestedArray.f64(2, delegate.defaultSize);
  const angvel = TypedArray.f64(delegate.defaultSize);
  return {
    position,
    rotation,
    linvel,
    angvel,
    addComponent(world: World, eid: EntityId) {
      addComponent(world, eid, RigidBodyState);
      position.getBatch(eid).fill(0);
      linvel.getBatch(eid).fill(0);
      angvel[eid] = 0;
      rotation[eid] = 0;
    },
    update(
      eid: number,
      translation: { x: number; y: number },
      rot: number,
      lv: { x: number; y: number },
      av: number,
    ) {
      position.set(eid, 0, translation.x);
      position.set(eid, 1, translation.y);
      rotation[eid] = rot;
      linvel.set(eid, 0, lv.x);
      linvel.set(eid, 1, lv.y);
      angvel[eid] = av;
    },
  };
});
