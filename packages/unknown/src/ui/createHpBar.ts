/**
 * createHpBar — the player's health readout: a thin bar pinned to the canvas's
 * bottom-left corner. Pure presentation — `setHealth(frac)` (0..1) drives the
 * fill width and its colour (green → red as it drops). The caller feeds it
 * `getTankHealth(playerEid)` each frame.
 *
 * Lives in `#screen` (which is shrink-wrapped to the canvas), so `absolute` +
 * left/bottom anchors it to the canvas corner in pure CSS.
 */

const CANVAS_INSET = 8; // gap (px) from the canvas's bottom-left corner
const BAR_WIDTH = 160;
const BAR_HEIGHT = 14;

export type HpBar = {
  element: HTMLElement;
  setHealth(frac: number): void;
  dispose(): void;
};

export function createHpBar(canvas: HTMLCanvasElement): HpBar {
  const parent = canvas.parentElement ?? document.body;

  const track = document.createElement("div");
  Object.assign(track.style, {
    position: "absolute",
    left: `${CANVAS_INSET}px`,
    bottom: `${CANVAS_INSET}px`,
    width: `${BAR_WIDTH}px`,
    height: `${BAR_HEIGHT}px`,
    border: "1px solid rgba(255, 255, 255, 0.35)",
    background: "rgba(0, 0, 0, 0.5)",
    zIndex: "10",
    overflow: "hidden",
    userSelect: "none",
  });

  const fill = document.createElement("div");
  Object.assign(fill.style, {
    height: "100%",
    width: "100%",
    transition: "width 120ms linear",
  });
  track.appendChild(fill);

  function setHealth(frac: number): void {
    const h = Math.max(0, Math.min(1, frac));
    fill.style.width = `${h * 100}%`;
    fill.style.background = `hsl(${h * 120}, 65%, 28%)`; // 0 = red, 1 = green (darkened)
  }

  setHealth(1);
  parent.appendChild(track);

  return {
    element: track,
    setHealth,
    dispose: () => track.remove(),
  };
}
