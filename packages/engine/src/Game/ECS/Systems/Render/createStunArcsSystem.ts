import { addEntity, entityExists, hasComponent, Not, query, removeComponent } from "bitecs";
import { GameDI } from "../../../DI/GameDI.ts";
import { RenderDI } from "../../../DI/RenderDI.ts";
import { getGameComponents } from "../../createGameWorld.ts";
import { addTransformComponents } from "renderer/src/ECS/Components/Transform.ts";
import { ShapeKind } from "renderer/src/ECS/Components/Shape.ts";
import { VFXType } from "../../Components/VFX.ts";
import { EmpStunConfig } from "../../../Config/weapons.ts";
import { EmpVfxConfig } from "../../../Config/vfx.ts";
import { scheduleRemoveEntity } from "../../Utils/typicalRemoveEntity.ts";
import { seedMultiplier } from "./VFX/createDrawVFXSystem.ts";

function fract(x: number): number {
  return x - Math.floor(x);
}

/**
 * JS mirror of the shader's `hash21` (vfx.shader.ts WGSL_UTILS), fed the same
 * `(slice, seed)` the bolts' strobe gate uses, so the ground glow flickers in
 * sync with the lightning re-strikes.
 */
function hash21(x: number, y: number): number {
  let p3x = fract(x * 0.1031);
  let p3y = fract(y * 0.103);
  let p3z = fract(x * 0.0973);
  // No intermediate fract — the WGSL does `p3 += dot(...)` and only wraps the
  // final product (fract of a product is not invariant under wrapping factors).
  const d = p3x * (p3y + 19.19) + p3y * (p3z + 19.19) + p3z * (p3x + 19.19);
  p3x += d;
  p3y += d;
  p3z += d;
  return fract((p3x + p3y) * p3z);
}

/**
 * Owns the crackling EmpOverlay overlay riding each stunned vehicle (render-only,
 * skipped headless). One persistent Parent-attached entity per stun: spawned
 * when a vehicle becomes `Stunned`, its `Progress` clock rewound every frame
 * to the true remaining stun (so refreshes restart the shader's tail fade),
 * its `LightEmitter` strobed in sync with the bolts, and torn down when the
 * stun expires. Vehicle death is NOT handled here — the overlay is registered
 * in the vehicle's `Children`, so the recursive destroy takes it down.
 *
 * Runs inside `spawnFrame` (structural adds at the sanctioned phase boundary).
 */
export function createStunArcsSystem({ world } = GameDI) {
  const {
    Stunned,
    StunArcs,
    Vehicle,
    Parent,
    Children,
    VFX,
    Progress,
    Shape,
    Color,
    LightEmitter,
  } = getGameComponents(world);
  const empOverlaySeedMult = seedMultiplier[VFXType.EmpOverlay];

  return (_delta: number) => {
    if (!RenderDI.enabled) return;

    const { color, lightIntensity, lightRadiusPx } = EmpVfxConfig.overlay;

    // Query A: freshly stunned vehicles → spawn the arc overlay.
    // Backwards: the final StunArcs.addComponent swap-removes the current
    // vehicle from this Not(StunArcs) query's dense array.
    const fresh = query(world, [Stunned, Vehicle, Not(StunArcs)]);
    for (let i = fresh.length - 1; i >= 0; i--) {
      const vehicleEid = fresh[i];
      const overlayEid = addEntity(world);

      // Identity LocalTransform + Parent and no RigidBodyRef → the
      // attached-transform system glues the overlay to the hull every frame.
      addTransformComponents(world, overlayEid);

      if (!hasComponent(world, vehicleEid, Children)) {
        Children.addComponent(world, vehicleEid);
      }
      Parent.addComponent(world, overlayEid, vehicleEid);
      // Death-path cleanup: the vehicle's recursive destroy walks Children,
      // so registering the overlay here is what takes it down with the tank.
      Children.addChildren(vehicleEid, overlayEid);

      VFX.addComponent(world, overlayEid, VFXType.EmpOverlay);
      Progress.addComponent(world, overlayEid, EmpStunConfig.durationMs);
      // Alpha-0 circle: invisible in the main pass, feeds the radiance
      // cascades emission pass (the spawnLightFlash precedent).
      Shape.addComponent(world, overlayEid, ShapeKind.Circle, lightRadiusPx);
      Color.addComponent(world, overlayEid, color[0], color[1], color[2], 0);
      LightEmitter.addComponent(world, overlayEid, lightIntensity);

      StunArcs.addComponent(world, vehicleEid, overlayEid);
    }

    // Query B: tracked overlays → rewind the shader clock + flicker the glow,
    // or tear down once the stun expired.
    // Backwards: removeComponent swap-removes inside the query's dense array.
    const tracked = query(world, [StunArcs]);
    for (let i = tracked.length - 1; i >= 0; i--) {
      const vehicleEid = tracked[i];
      const overlayEid = StunArcs.overlayEid.get(vehicleEid);
      const overlayAlive = overlayEid !== 0 && entityExists(world, overlayEid);

      if (overlayAlive && hasComponent(world, vehicleEid, Stunned)) {
        const remainingMs = Stunned.remainingMs.get(vehicleEid);
        Progress.age.set(overlayEid, Progress.maxAge.get(overlayEid) - remainingMs);

        const frac = Stunned.getRemainingFraction(vehicleEid);
        const slice = Math.floor(Progress.getProgress(overlayEid) * 24);
        const sliceHash = hash21(slice, (overlayEid * empOverlaySeedMult) % 1.0);
        LightEmitter.set$(overlayEid, lightIntensity * frac * (0.6 + 0.8 * sliceHash), 0);
      } else {
        if (overlayAlive) {
          scheduleRemoveEntity(overlayEid);
        }
        removeComponent(world, vehicleEid, StunArcs);
      }
    }
  };
}
