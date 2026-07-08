import { hasComponent, query } from "bitecs";
import { getRenderComponents, type RenderWorld } from "../../world.ts";

type VoxelLightSink = {
  setLights(data: Float32Array, count: number): void;
};

// Auto-discover every emitter from the ECS each frame → cone importance-sampling lights
// (UNCAPPED — the gather stores them in a runtime-sized storage buffer and round-robins
// aimedPerFrame of them per probe). count=0 → pure Fibonacci fill cones, so a scene with
// only the directional sun is a no-op here. center = transform translation (center-origin),
// radius = the sphere radius (Shape.values[0]) for emitter spheres, default 0.5.
export function createLightEmitterSystem(world: RenderWorld, voxel: VoxelLightSink) {
  const { LocalTransform, LightEmitter, Shape, Color } = getRenderComponents(world);
  // Interleaved emitter records, 8 floats each (x,y,z,radius, r,g,b,intensity) — the gather's
  // uLights storage layout (two vec4 per light). Grow-doubled, reused across frames.
  let data = new Float32Array(64 * 8);

  return function execLightEmitterSystem() {
    const ents = query(world, [LightEmitter, LocalTransform]);
    if (ents.length * 8 > data.length) {
      let cap = data.length;
      while (cap < ents.length * 8) cap *= 2;
      data = new Float32Array(cap);
    }
    for (let i = 0; i < ents.length; i++) {
      const id = ents[i];
      const o = i * 8;
      data[o + 0] = LocalTransform.matrix.get(id, 12);
      data[o + 1] = LocalTransform.matrix.get(id, 13);
      data[o + 2] = LocalTransform.matrix.get(id, 14);
      data[o + 3] = hasComponent(world, id, Shape) ? Shape.values.get(id, 0) || 0.5 : 0.5;
      data[o + 4] = Color.getR(id);
      data[o + 5] = Color.getG(id);
      data[o + 6] = Color.getB(id);
      data[o + 7] = LightEmitter.intensity[id];
    }
    voxel.setLights(data, ents.length);
  };
}
