import { hasComponent, query } from "bitecs";
import { getRenderComponents, type RenderWorld } from "../../world.ts";

type VoxelLightSink = {
  setLights(flat: Float32Array, count: number, colorsFlat: Float32Array): void;
};

// Auto-discover every emitter from the ECS each frame → cone importance-sampling lights
// (capped at 8 by voxel.setLights). count=0 → pure Fibonacci fill cones, so a scene with
// only the directional sun is a no-op here. center = transform translation (center-origin),
// radius = the sphere radius (Shape.values[0]) for emitter spheres, default 0.5.
export function createLightEmitterSystem(world: RenderWorld, voxel: VoxelLightSink) {
  const { LocalTransform, LightEmitter, Shape, Color } = getRenderComponents(world);
  // Preallocated, reused across frames: 8 lights × 4 floats (x,y,z,radius / r,g,b,intensity).
  const lights = new Float32Array(32);
  const colors = new Float32Array(32);

  return function execLightEmitterSystem() {
    lights.fill(0);
    colors.fill(0);
    const ents = query(world, [LightEmitter, LocalTransform]);
    let n = 0;
    for (let i = 0; i < ents.length && n < 8; i++) {
      const id = ents[i];
      const o = n * 4;
      lights[o + 0] = LocalTransform.matrix.get(id, 12);
      lights[o + 1] = LocalTransform.matrix.get(id, 13);
      lights[o + 2] = LocalTransform.matrix.get(id, 14);
      lights[o + 3] = hasComponent(world, id, Shape) ? Shape.values.get(id, 0) || 0.5 : 0.5;
      colors[o + 0] = Color.getR(id);
      colors[o + 1] = Color.getG(id);
      colors[o + 2] = Color.getB(id);
      colors[o + 3] = LightEmitter.intensity[id];
      n++;
    }
    voxel.setLights(lights, n, colors);
  };
}
