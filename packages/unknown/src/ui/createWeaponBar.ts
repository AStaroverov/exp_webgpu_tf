/**
 * createWeaponBar — the player's weapon selector: four pixel-art icon buttons
 * pinned to the bottom edge of the game canvas (normal shell / flame / frost /
 * EMP), each labelled with its hotkey number 1–4. Pure DOM + tiny hand-drawn
 * sprites rendered onto a 1px-per-cell canvas and scaled up with
 * `image-rendering: pixelated`, so they stay crisp and blocky at any size.
 *
 * Selection works by click OR by pressing 1–4. It owns no game state — choosing a
 * weapon highlights it and calls `onSelect`; the caller wires that to
 * `setTankWeapon`. Mousedown on the bar is swallowed so clicking a button over
 * the canvas doesn't also trip the fire control (which listens on `window`).
 */

import type { PlayerWeapon } from "../Game/ECS/Entities/Tank/setTankWeapon.ts";

/** A pixel sprite: a char→color palette and rows of those chars (`.` = transparent). */
type Sprite = { palette: Record<string, string>; rows: string[] };

const NORMAL: Sprite = {
  palette: { t: "#e6b34d", b: "#c2cad6", d: "#8b94a3" },
  rows: [
    ".....t.....",
    "....ttt....",
    "....ttt....",
    "...bbbbb...",
    "...bbbbb...",
    "...bbbbb...",
    "...bbbbb...",
    "...bbbbb...",
    "...ddddd...",
    "...d.d.d...",
    "...ddddd...",
  ],
};

const FLAME: Sprite = {
  palette: { r: "#e8412b", o: "#ff8a2b", y: "#ffd24a" },
  rows: [
    ".....r.....",
    ".....o.....",
    "....roo....",
    "....roo....",
    "...rooyo...",
    "...rooyo...",
    "..rooyyoo..",
    "..rooyyoo..",
    "..roooooo..",
    "...roooo...",
    "....rrr....",
  ],
};

const FROST: Sprite = {
  palette: { b: "#3aa0e6", w: "#cdeeff" },
  rows: [
    "b....b....b",
    ".b...b...b.",
    "..b..b..b..",
    "...b.b.b...",
    "....bbb....",
    "bbbbbwbbbbb",
    "....bbb....",
    "...b.b.b...",
    "..b..b..b..",
    ".b...b...b.",
    "b....b....b",
  ],
};

const EMP: Sprite = {
  palette: { e: "#7ad0ff", w: "#ffffff" },
  rows: [
    ".......ee..",
    "......ee...",
    ".....ee....",
    "....eeww...",
    "...eewwee..",
    ".....ee....",
    "....ee.....",
    "...ee......",
    "..eee......",
    ".ee........",
    "...........",
  ],
};

/** Display order of the bar, left → right — index + 1 is the hotkey number. */
const WEAPONS: ReadonlyArray<{ weapon: PlayerWeapon; sprite: Sprite }> = [
  { weapon: "normal", sprite: NORMAL },
  { weapon: "flame", sprite: FLAME },
  { weapon: "frost", sprite: FROST },
  { weapon: "emp", sprite: EMP },
];

const PIXEL_SCALE = 2; // each sprite cell → this many CSS px
const CANVAS_INSET = 8; // gap (px) from the canvas's bottom-right corner

/** Render a sprite onto a 1px-per-cell canvas scaled up with pixelated rendering. */
function renderSprite({ palette, rows }: Sprite): HTMLCanvasElement {
  const h = rows.length;
  const w = rows[0].length;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  for (let y = 0; y < h; y++) {
    const row = rows[y];
    for (let x = 0; x < row.length; x++) {
      const color = palette[row[x]];
      if (!color) continue;
      ctx.fillStyle = color;
      ctx.fillRect(x, y, 1, 1);
    }
  }
  canvas.style.width = `${w * PIXEL_SCALE}px`;
  canvas.style.height = `${h * PIXEL_SCALE}px`;
  canvas.style.imageRendering = "pixelated";
  return canvas;
}

export type WeaponBar = {
  element: HTMLElement;
  /** Highlight the given weapon as active (no callback). */
  setActive(weapon: PlayerWeapon): void;
  dispose(): void;
};

export function createWeaponBar(
  canvas: HTMLCanvasElement,
  opts: { onSelect: (weapon: PlayerWeapon) => void; initial: PlayerWeapon },
): WeaponBar {
  // `#screen` is shrink-wrapped to the canvas, so absolute + right/bottom pins the
  // bar to the canvas's bottom-right corner in pure CSS (no measuring / resize).
  const parent = canvas.parentElement ?? document.body;

  const bar = document.createElement("div");
  Object.assign(bar.style, {
    position: "absolute",
    right: `${CANVAS_INSET}px`,
    bottom: `${CANVAS_INSET}px`,
    display: "flex",
    gap: "5px",
    padding: "4px",
    borderRadius: "6px",
    background: "rgba(0, 0, 0, 0.45)",
    zIndex: "10",
    userSelect: "none",
  });
  // Don't let a button press double as a fire click (the fire control listens on window).
  bar.addEventListener("mousedown", (e) => e.stopPropagation());

  const buttons = new Map<PlayerWeapon, HTMLButtonElement>();

  WEAPONS.forEach(({ weapon, sprite }, i) => {
    const btn = document.createElement("button");
    Object.assign(btn.style, {
      position: "relative", // anchor for the absolutely-placed hotkey number
      display: "block",
      padding: "3px",
      border: "1px solid transparent",
      borderRadius: "4px",
      background: "rgba(255, 255, 255, 0.06)",
      color: "#dfe6f0",
      font: "bold 9px/1 system-ui, sans-serif",
      cursor: "pointer",
    });
    btn.appendChild(renderSprite(sprite));
    const num = document.createElement("span");
    num.textContent = String(i + 1);
    Object.assign(num.style, {
      position: "absolute",
      top: "1px",
      left: "2px",
      textShadow: "0 0 2px #000, 0 0 2px #000", // keep it legible over the icon
    });
    btn.appendChild(num);
    btn.addEventListener("click", () => select(weapon));
    buttons.set(weapon, btn);
    bar.appendChild(btn);
  });

  function setActive(weapon: PlayerWeapon): void {
    for (const [w, btn] of buttons) {
      const active = w === weapon;
      btn.style.borderColor = active ? "#ffffff" : "transparent";
      btn.style.background = active ? "rgba(255, 255, 255, 0.18)" : "rgba(255, 255, 255, 0.06)";
    }
  }

  function select(weapon: PlayerWeapon): void {
    setActive(weapon);
    opts.onSelect(weapon);
  }

  // Hotkeys 1–4 → the weapon at that index.
  const onKeyDown = (e: KeyboardEvent) => {
    const idx = Number(e.key) - 1;
    if (idx >= 0 && idx < WEAPONS.length) select(WEAPONS[idx].weapon);
  };
  window.addEventListener("keydown", onKeyDown);

  setActive(opts.initial);
  parent.appendChild(bar);

  return {
    element: bar,
    setActive,
    dispose: () => {
      window.removeEventListener("keydown", onKeyDown);
      bar.remove();
    },
  };
}
