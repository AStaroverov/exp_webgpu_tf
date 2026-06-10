import { query } from "bitecs";
import { GameDI } from "../../../DI/GameDI.ts";
import { getGameComponents } from "../../createGameWorld.ts";

const INITIAL_ALPHA = 0.4;

export function createUpdateTreadMarksSystem({ world } = GameDI) {
  const { Color, TreadMark, Progress } = getGameComponents(world);

  return () => {
    const treadMarkEids = query(world, [TreadMark, Progress]);

    for (const eid of treadMarkEids) {
      const progress = Progress.getProgress(eid);
      const alpha = INITIAL_ALPHA * (1 - progress);

      if (alpha - Color.getA(eid) > 0.05) {
        Color.setA$(eid, alpha);
      }
    }
  };
}
