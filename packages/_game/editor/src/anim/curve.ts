// Phase-keyed scalar curve shared by procedural stances: a list of {at, v} control points sampled
// with smoothstep easing between neighbours. (Clips use Catmull-Rom over full poses; this is the
// lightweight 1D analogue for hand-authored procedural drivers like a draw or lunge amount.)
export type Key = { at: number; v: number };

export function smoothstep(x: number): number {
  return x * x * (3 - 2 * x);
}

export function sampleKeys(keys: Key[], p: number): number {
  for (let i = 0; i < keys.length - 1; i++) {
    const a = keys[i];
    const b = keys[i + 1];
    if (p <= b.at) return a.v + (b.v - a.v) * smoothstep((p - a.at) / (b.at - a.at));
  }
  return keys[keys.length - 1].v;
}
