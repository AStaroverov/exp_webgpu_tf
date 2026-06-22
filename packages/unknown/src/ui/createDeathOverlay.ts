/**
 * createDeathOverlay — a "You died / Restart" panel shown over the canvas when the
 * player's tank is destroyed. Pure DOM, hidden until `show()`. The Restart button
 * calls `onRestart` (the caller wires it to a full page reload — the simplest,
 * leak-free way to rebuild the GPU/tf/world/timers from scratch).
 *
 * Lives in `#screen` (shrink-wrapped to the canvas) and covers it via inset 0.
 */

export type DeathOverlay = {
  element: HTMLElement;
  show(): void;
  hide(): void;
  dispose(): void;
};

export function createDeathOverlay(
  canvas: HTMLCanvasElement,
  opts: { onRestart: () => void },
): DeathOverlay {
  const parent = canvas.parentElement ?? document.body;

  const overlay = document.createElement("div");
  Object.assign(overlay.style, {
    position: "absolute",
    inset: "0",
    display: "none",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    gap: "16px",
    background: "rgba(0, 0, 0, 0.6)",
    zIndex: "20",
    userSelect: "none",
  });

  const title = document.createElement("div");
  title.textContent = "DESTROYED";
  Object.assign(title.style, {
    color: "#e8e8ee",
    font: "bold 28px/1 system-ui, sans-serif",
    letterSpacing: "2px",
    textShadow: "0 0 6px #000",
  });
  overlay.appendChild(title);

  const button = document.createElement("button");
  button.textContent = "Restart";
  Object.assign(button.style, {
    padding: "10px 22px",
    border: "2px solid #ffffff",
    borderRadius: "6px",
    background: "rgba(255, 255, 255, 0.12)",
    color: "#ffffff",
    font: "bold 15px/1 system-ui, sans-serif",
    cursor: "pointer",
  });
  button.addEventListener("click", opts.onRestart);
  overlay.appendChild(button);

  parent.appendChild(overlay);

  return {
    element: overlay,
    show: () => {
      overlay.style.display = "flex";
    },
    hide: () => {
      overlay.style.display = "none";
    },
    dispose: () => overlay.remove(),
  };
}
