import type { EntityId } from "bitecs";
import { hasComponent, query, removeComponent } from "bitecs";
import { GameDI } from "../../../DI/GameDI.ts";
import { RenderDI } from "../../../DI/RenderDI.ts";
import { getGameComponents } from "../../createGameWorld.ts";
import { StreamCaliberConfig } from "../../../Config/weapons.ts";
import { EmpVfxConfig } from "../../../Config/vfx.ts";
import { DamageKind } from "../../Components/Damagable.ts";
import { findVehicleEidByPartEid } from "../../Utils/findPartVehicle.ts";
import { getSlotFillerEid, isSlot } from "../../Utils/SlotUtils.ts";
import { lerp, min } from "../../../../../../../lib/math.ts";

const FIRE = StreamCaliberConfig.find((c) => c.kind === DamageKind.Fire)!;
const FROST = StreamCaliberConfig.find((c) => c.kind === DamageKind.Frost)!;

/**
 * Single owner of all damage-status recoloring (render-only, skipped headless).
 * Each frame a part's color is blended from its `OriginalColor` snapshot toward
 * the effect tint, proportional to the effect magnitude: a Fire-`Dot` part by
 * `dps / fire dps`, every part of a `Stunned` vehicle by `remainingMs /
 * durationMs`, every part of a `Slowed` vehicle by `slowMul`
 * (Fire > Emp > Frost per part). The snapshot — the slot color after brightness
 * jitter — is taken lazily on the first tint and restored once neither applies.
 */
export function createTintSystem({ world } = GameDI) {
  const { Dot, Slowed, Stunned, Color, OriginalColor, Children } = getGameComponents(world);

  // Blend the part's color = lerp(original, tint, intensity ∈ [0,1]).
  const applyTint = (
    partEid: EntityId,
    tint: readonly [number, number, number],
    intensity: number,
  ) => {
    if (!hasComponent(world, partEid, OriginalColor)) {
      OriginalColor.addComponent(
        world,
        partEid,
        Color.getR(partEid),
        Color.getG(partEid),
        Color.getB(partEid),
      );
    }
    Color.set$(
      partEid,
      lerp(OriginalColor.r.get(partEid), tint[0], intensity),
      lerp(OriginalColor.g.get(partEid), tint[1], intensity),
      lerp(OriginalColor.b.get(partEid), tint[2], intensity),
      Color.getA(partEid),
    );
  };

  const isFireDot = (partEid: EntityId) =>
    hasComponent(world, partEid, Dot) && Dot.kind.get(partEid) === DamageKind.Fire;

  // Walks the slots under a vehicle/turret and tints their filler parts.
  const tintSlotParts = (
    parentEid: EntityId,
    tint: readonly [number, number, number],
    intensity: number,
  ) => {
    const childCount = Children.entitiesCount.get(parentEid);

    for (let i = 0; i < childCount; i++) {
      const childEid = Children.entitiesIds.get(parentEid, i);

      if (!isSlot(childEid)) {
        // Non-slot child with its own slots (the turret/gun) — descend.
        if (hasComponent(world, childEid, Children)) tintSlotParts(childEid, tint, intensity);
        continue;
      }

      const partEid = getSlotFillerEid(childEid);
      if (partEid === 0) continue;
      if (isFireDot(partEid)) continue; // Fire wins
      applyTint(partEid, tint, intensity);
    }
  };

  return (_delta: number) => {
    if (!RenderDI.enabled) return;

    const dotEids = query(world, [Dot, Color]);
    for (let i = 0; i < dotEids.length; i++) {
      const partEid = dotEids[i];
      if (Dot.kind.get(partEid) !== DamageKind.Fire) continue;
      applyTint(partEid, FIRE.tint, min(1, Dot.dps.get(partEid) / FIRE.dot.dps));
    }

    const stunnedEids = query(world, [Stunned, Children]);
    for (let i = 0; i < stunnedEids.length; i++) {
      const vehicleEid = stunnedEids[i];
      tintSlotParts(vehicleEid, EmpVfxConfig.tint, Stunned.getRemainingFraction(vehicleEid));
    }

    const slowedEids = query(world, [Slowed, Children]);
    for (let i = 0; i < slowedEids.length; i++) {
      const vehicleEid = slowedEids[i];
      // Emp wins over Frost — the whole subtree belongs to this vehicle.
      if (hasComponent(world, vehicleEid, Stunned)) continue;
      tintSlotParts(vehicleEid, FROST.tint, Slowed.slowMul.get(vehicleEid));
    }

    // Revert: backwards — removeComponent swap-removes inside the query's dense array.
    const recoloredEids = query(world, [OriginalColor, Color]);
    for (let i = recoloredEids.length - 1; i >= 0; i--) {
      const partEid = recoloredEids[i];

      if (isFireDot(partEid)) continue;
      // Torn-off debris resolves to no vehicle → not slowed/stunned → revert.
      const vehicleEid = findVehicleEidByPartEid(partEid);
      if (
        vehicleEid !== undefined &&
        (hasComponent(world, vehicleEid, Slowed) || hasComponent(world, vehicleEid, Stunned))
      )
        continue;

      Color.set$(
        partEid,
        OriginalColor.r.get(partEid),
        OriginalColor.g.get(partEid),
        OriginalColor.b.get(partEid),
        Color.getA(partEid),
      );
      removeComponent(world, partEid, OriginalColor);
    }
  };
}
