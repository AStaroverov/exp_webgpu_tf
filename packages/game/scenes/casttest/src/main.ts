import { createEngine } from "../../../../engine/src/createEngine.ts";
import {
  createEntityId,
  getEngineComponents,
  type EngineWorld,
} from "../../../../engine/src/ECS/createEngineWorld.ts";
import { createGround } from "../../../../engine/src/ECS/Entities/RigidShapes.ts";
import { ColliderDebug } from "../../../../engine/src/ECS/Systems/createColliderDebugSystem.ts";
import {
  addTransformComponents,
  applyMatrixTranslate,
} from "../../../../renderer/src/ECS/Components/Transform.ts";

// Headless smoke test for the CAST_SHAPE query channel (no canvas / no WebGPU): a
// blade-like ShapeCaster segment at the origin pointing +X must report the sphere
// body it passes through, must NOT report the far sphere, and must NOT report the
// excluded body. Exposes `window.__castResult` for the driving script.

declare global {
  interface Window {
    __castResult?: { pass: boolean; detail: string };
  }
}

function report(pass: boolean, detail: string): void {
  document.getElementById("log")!.textContent = `${pass ? "PASS" : "FAIL"}\n${detail}`;
  window.__castResult = { pass, detail };
}

async function main(): Promise<void> {
  const engine = await createEngine({});
  const world = engine.world as EngineWorld;
  const { Children, RigidBodyState, ShapeCaster } = getEngineComponents(world);

  createGround(world, { size: 40, thickness: 1, z: 0, color: [0.2, 0.2, 0.2, 1] });

  const makeBall = (x: number, y: number): number => {
    const eid = createEntityId(world);
    RigidBodyState.addComponent(world, eid, {
      kind: "sphere",
      bodyType: "dynamic",
      position: { x, y, z: 0.6 },
      radius: 0.6,
    });
    return eid;
  };
  const target = makeBall(2.5, 0); // on the blade line → must be hit
  const far = makeBall(0, 6); // far away → must NOT be hit
  const self = makeBall(0.3, 0); // overlaps the hilt → excluded, must NOT be hit

  const caster = createEntityId(world);
  addTransformComponents(world, caster);
  Children.addComponent(world, caster);
  Children.addChild(engine.sceneRoot, caster);
  applyMatrixTranslate(
    getEngineComponents(world).LocalTransform.matrix.getBatch(caster),
    0,
    0,
    0.6,
  );
  // Blade in the caster's LOCAL space: hilt (0.2,0,0) → tip (4,0,0) along +X.
  ShapeCaster.addComponent(world, caster, {
    localFrom: [0.2, 0, 0],
    localTo: [4, 0, 0],
    radius: 0.2,
    excludeEid: self,
  });

  // Collider debug ghosts: 4 bodies (ground + 3 balls) + 1 caster cylinder land
  // under sceneRoot next to the caster entity itself.
  ColliderDebug.enabled = true;
  const baseChildren = Children.entitiesCount.get(engine.sceneRoot);

  // Find a sceneRoot child whose world X/Z translation matches, excluding known eids.
  function findGhostAt(x: number, z: number, tolerance: number): number {
    const { LocalTransform } = getEngineComponents(world);
    const count = Children.entitiesCount.get(engine.sceneRoot);
    for (let i = 0; i < count; i++) {
      const eid = Children.entitiesIds.get(engine.sceneRoot, i);
      if (eid === caster) continue;
      const m = LocalTransform.matrix.getBatch(eid);
      if (Math.abs(m[12] - x) < tolerance && Math.abs(m[14] - z) < tolerance) return eid;
    }
    return -1;
  }

  const seen = new Set<number>();
  let frames = 0;
  function loop(): void {
    engine.tick(16.6667 / 1000);
    const count = ShapeCaster.getHitCount(caster);
    for (let i = 0; i < count; i++) seen.add(ShapeCaster.getHit(caster, i));
    if (++frames < 60) {
      requestAnimationFrame(loop);
      return;
    }

    const castPass = seen.has(target) && !seen.has(far) && !seen.has(self);

    const ghostChildren = Children.entitiesCount.get(engine.sceneRoot) - baseChildren;
    const targetGhost = findGhostAt(2.5, 0.6, 0.12); // ball body ghost at its pose
    const casterGhost = findGhostAt(2.1, 0.6, 0.12); // blade cylinder at segment midpoint
    ColliderDebug.enabled = false;
    engine.tick(16.6667 / 1000);
    const cleaned = Children.entitiesCount.get(engine.sceneRoot) === baseChildren;
    const debugPass = ghostChildren === 5 && targetGhost >= 0 && casterGhost >= 0 && cleaned;

    const detail =
      `cast: seen=[${[...seen].join(",")}] target=${target} far=${far} self=${self}\n` +
      `debug: ghosts=${ghostChildren}/5 targetGhost=${targetGhost} casterGhost=${casterGhost} ` +
      `cleaned=${cleaned}`;
    report(castPass && debugPass, detail);
  }
  requestAnimationFrame(loop);
}

main().catch((err) => {
  report(false, `crash: ${(err as Error)?.stack ?? String(err)}`);
});
