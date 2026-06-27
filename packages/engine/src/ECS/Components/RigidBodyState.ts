import { addComponent, World } from "bitecs";
import { delegate } from "../../../../renderer3d_2/src/delegate.ts";
import { NestedArray } from "../../../../renderer3d_2/src/utils.ts";
import { defineComponent } from "../../../../renderer3d_2/src/ECS/utils.ts";

// Per-frame physics snapshot in 3D. The single biggest 2D→3D delta from the
// unknown package: position is [x,y,z], rotation is a QUATERNION [x,y,z,w]
// (2D stored a scalar angle), and linvel/angvel are [x,y,z].
export const createRigidBodyStateComponent = defineComponent((RigidBodyState) => {
  const position = NestedArray.f64(3, delegate.defaultSize);
  const rotation = NestedArray.f64(4, delegate.defaultSize);
  const linvel = NestedArray.f64(3, delegate.defaultSize);
  const angvel = NestedArray.f64(3, delegate.defaultSize);
  return {
    position,
    rotation,
    linvel,
    angvel,
    addComponent(world: World, eid: number) {
      addComponent(world, eid, RigidBodyState);
      position.getBatch(eid).fill(0);
      rotation.getBatch(eid).fill(0);
      // Identity quaternion: w = 1 so a never-synced (fixed) body has a valid rotation.
      rotation.set(eid, 3, 1);
      linvel.getBatch(eid).fill(0);
      angvel.getBatch(eid).fill(0);
    },
    update(
      eid: number,
      px: number,
      py: number,
      pz: number,
      qx: number,
      qy: number,
      qz: number,
      qw: number,
      lx: number,
      ly: number,
      lz: number,
      ax: number,
      ay: number,
      az: number,
    ) {
      position.set(eid, 0, px).set(eid, 1, py).set(eid, 2, pz);
      rotation.set(eid, 0, qx).set(eid, 1, qy).set(eid, 2, qz).set(eid, 3, qw);
      linvel.set(eid, 0, lx).set(eid, 1, ly).set(eid, 2, lz);
      angvel.set(eid, 0, ax).set(eid, 1, ay).set(eid, 2, az);
    },
  };
});
