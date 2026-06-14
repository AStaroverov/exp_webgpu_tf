import { ActiveCollisionTypes, ActiveEvents, ColliderDesc } from "@dimforge/rapier2d-simd";
import { GameDI } from "../DI/GameDI.ts";
import { BodyOptions, createBody } from "./createBody.ts";
import { PhysicalWorld } from "./initPhysicalWorld.ts";
import { CollisionGroupConfig, TANK_PARTS_MASK } from "../Config/index.ts";
import { getGameComponents } from "../ECS/createGameWorld.ts";

export const CollisionGroup = {
  ...CollisionGroupConfig,
  // Legacy naming compatibility
  VEHICALE_BASE: CollisionGroupConfig.VEHICLE_BASE,
  VEHICALE_HULL_PARTS: CollisionGroupConfig.VEHICLE_HULL_PARTS,
  TANK_PARTS: TANK_PARTS_MASK,
} as const;

type CollisionGroupValue = (typeof CollisionGroup)[keyof typeof CollisionGroup];

type CommonRigidOptions = BodyOptions & {
  enabled?: boolean;
  density?: number;
  belongsCollisionGroup?: 0 | CollisionGroupValue;
  interactsCollisionGroup?: 0 | CollisionGroupValue;
  belongsSolverGroup?: 0 | CollisionGroupValue;
  interactsSolverGroup?: 0 | CollisionGroupValue;
  collisionEvent?: ActiveEvents;
  activeCollisionTypes?: ActiveCollisionTypes;
  /** If true, collider is a sensor (detects intersections but no physics response) */
  sensor?: boolean;
};

export function createCollisionGroups(belongs: 0 | number, interacts: 0 | number) {
  return (belongs << 16) | interacts;
}

function prepareColliderDesc(shape: ColliderDesc, o: CommonRigidOptions): ColliderDesc {
  return shape
    .setEnabled(o.enabled ?? true)
    .setDensity(o.density ?? 0)
    .setSensor(o.sensor ?? false)
    .setCollisionGroups(
      createCollisionGroups(
        o.belongsCollisionGroup ?? CollisionGroup.ALL,
        o.interactsCollisionGroup ?? CollisionGroup.ALL,
      ),
    )
    .setSolverGroups(
      createCollisionGroups(
        o.belongsSolverGroup ?? CollisionGroup.ALL,
        o.interactsSolverGroup ?? CollisionGroup.ALL,
      ),
    )
    .setActiveEvents(o.collisionEvent ?? ActiveEvents.NONE)
    .setActiveCollisionTypes(o.activeCollisionTypes ?? ActiveCollisionTypes.DEFAULT);
}

export function createRigidRectangle(
  o: CommonRigidOptions & {
    width: number;
    height: number;
  },
  { physicalWorld }: { physicalWorld: PhysicalWorld } = GameDI,
) {
  const body = createBody(o, { physicalWorld });
  const colliderDesc = prepareColliderDesc(ColliderDesc.cuboid(o.width / 2, o.height / 2), o);
  physicalWorld.createCollider(colliderDesc, body);

  return body.handle;
}

export function createRigidCircle(
  o: CommonRigidOptions & {
    radius: number;
  },
  { physicalWorld }: { physicalWorld: PhysicalWorld } = GameDI,
) {
  const body = createBody(o, { physicalWorld });
  const colliderDesc = prepareColliderDesc(ColliderDesc.ball(o.radius / 2), o);
  physicalWorld.createCollider(colliderDesc, body);

  return body.handle;
}

export function attachRigidRectangleCollider(
  ownerPid: number,
  o: CommonRigidOptions & { width: number; height: number; offsetX: number; offsetY: number },
  { physicalWorld }: { physicalWorld: PhysicalWorld } = GameDI,
): number {
  const ownerBody = physicalWorld.getRigidBody(ownerPid);
  const colliderDesc = prepareColliderDesc(
    ColliderDesc.cuboid(o.width / 2, o.height / 2),
    o,
  ).setTranslation(o.offsetX, o.offsetY);
  const collider = physicalWorld.createCollider(colliderDesc, ownerBody);
  return collider.handle;
}

export function removeRigidShape(eid: number, { world, physicalWorld } = GameDI) {
  const { RigidBodyRef } = getGameComponents(world);
  const pid = RigidBodyRef.id[eid];
  const body = physicalWorld.getRigidBody(pid);
  if (body) physicalWorld.removeRigidBody(body);
}
