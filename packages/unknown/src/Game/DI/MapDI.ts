/**
 * MapDI — module-level singleton holding the active hex grid, mirroring the
 * GameDI/RenderDI/SoundDI pattern. Set during `createGame`, cleared on destroy.
 */

import { HexGrid } from "../Map/HexGrid.ts";

export const MapDI: {
  grid: HexGrid;
} = {
  grid: null as any,
};
