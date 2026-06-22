/**
 * createScoreCounter — the player's score readout, pinned to the canvas's
 * top-left corner. Pure presentation — `setScore(points)` drives the text. The
 * caller feeds it the player's `Score` value each frame.
 *
 * Lives in `#screen` (which is shrink-wrapped to the canvas), so `absolute` +
 * left/top anchors it to the canvas corner in pure CSS.
 */

const CANVAS_INSET = 8; // gap (px) from the canvas's top-left corner

export type ScoreCounter = {
  element: HTMLElement;
  setScore(points: number): void;
  dispose(): void;
};

export function createScoreCounter(canvas: HTMLCanvasElement): ScoreCounter {
  const parent = canvas.parentElement ?? document.body;

  const label = document.createElement("div");
  Object.assign(label.style, {
    position: "absolute",
    left: `${CANVAS_INSET}px`,
    top: `${CANVAS_INSET}px`,
    color: "#e8e8ee",
    font: "bold 18px/1 system-ui, sans-serif",
    letterSpacing: "1px",
    textShadow: "0 0 4px #000, 0 1px 2px #000",
    zIndex: "10",
    userSelect: "none",
    pointerEvents: "none",
  });

  let shown = -1;
  function setScore(points: number): void {
    const value = Math.floor(points);
    if (value === shown) return; // skip DOM writes when nothing changed
    shown = value;
    label.textContent = `SCORE ${value}`;
  }

  setScore(0);
  parent.appendChild(label);

  return {
    element: label,
    setScore,
    dispose: () => label.remove(),
  };
}
