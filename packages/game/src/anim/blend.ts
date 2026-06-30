// A weighted animation layer: an eased blend weight (0 = off, 1 = active) plus its own looping clock.
// Each frame you push it toward a target; the weight chases the target with an exponential ease. While
// the weight is ~0 the layer is SKIPPED — `apply` is not called, so the bone keeps whatever the base
// pose left it (no need to re-write rest every frame). Otherwise it calls `apply(phase, weight)`, where
// phase is the clock wrapped into [0,1) over `duration`. Both procedural stances and clip players plug
// in: the apply callback does the posing and reads `weight` to blend onto the current pose.
const OFF_EPS = 1e-3;

export function makeBlendLayer(
  duration: number,
  apply: (phase: number, weight: number) => void,
  easeRate = 8,
): (delta: number, target: number) => void {
  let weight = 0;
  let clock = 0;
  return (delta, target) => {
    weight += (target - weight) * (1 - Math.exp(-delta * easeRate));
    clock += delta;
    if (weight < OFF_EPS) return;
    apply((clock % duration) / duration, weight);
  };
}
