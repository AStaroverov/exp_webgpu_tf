# `engine` package — implementation plan

A generic integration layer that fuses **Rapier 3D physics**
(`@dimforge/rapier3d-simd`) with the **renderer3d_2** (2.5D SDF) renderer over a
**bitecs** ECS. It recreates the same physics↔render seam the `unknown` package has,
but in 3D and with **zero game/business logic**. The deliverable is a runnable demo:
gravity drops boxes and spheres onto a ground plane, lit by the VCT sun, viewed by a
tilted orbiting orthographic camera.

renderer3d_2 is consumed by **relative path** (`../../renderer3d_2/src/...`), exactly
as `unknown` imports `renderer`. No build/bundle step for the renderer; we import its
`.ts` directly through Vite.

---

## 1. Overview & goals

**What it is.** `engine` is the thin, reusable glue between three subsystems that are
each already complete:

- **Physics** — Rapier 3D (`@dimforge/rapier3d-simd`), full 6-DOF rigid bodies.
- **Render** — renderer3d_2, a **2.5D SDF impostor** renderer with Voxel Cone Tracing
  GI and a directional sun.
- **ECS** — bitecs (SoA typed-array columns indexed by `eid`), the shared substrate.

The engine owns only: DI singletons, the physics world bootstrap, the bridging
components (`RigidBodyRef`, `RigidBodyState3D`), the two sync systems, generic body
factories, a `createEngine` frame assembler, and a `demo.ts`. **It contains no game
concepts** (no weapons, teams, AI, sound) — it is the 3D analogue of `unknown`'s
physics↔render spine, lifted out clean.

**The 2.5D-vs-3D rotation reality (state plainly).** The renderer's SDF pass
(`packages/renderer3d_2/src/ECS/Systems/SDFSystem/sdf.shader.ts` +
`sceneSDF.wgsl.ts`) is a **2.5D impostor renderer by construction**. From a per-instance
`mat4` it reads only the translation (`transform[3].xyz`) and a single **yaw** angle
derived as `atan2(transform[0].y, transform[0].x)`. The full upper-left 3×3 basis
(`transform[1]`, `transform[2]`) is never used; extrusion is welded to world-Z in
`sd_shape3d`. Therefore, mapping physics → render:

| Physics quantity | Renders faithfully? |
|---|---|
| **Translation (x, y, z)** | **Always.** `transform[3].xyz`, full 3D. |
| **Yaw (rotation about Z)** | **Yes.** Extracted via `atan2`. |
| **Pitch / roll (about X or Y)** | **No.** Silently dropped — a tumbling box renders upright. |
| **Spheres (kind 6)** | Rotation-**invariant** (`length(p) - r`): rotation is unobservable, so always "correct". |
| **Scale** | **Ignored.** Matrix scale is normalized away by `atan2`. Size comes only from `Shape.values` (footprint) + `Height` (extrusion). |

The honest position of this engine: **translation is fully 3D; rotation is faithful
only for yaw and for spheres.** Boxes that tumble about X/Y will render as if upright.
This is acceptable for the demo (boxes settle roughly upright; spheres are
rotation-invariant) and is documented as a **renderer limitation, not a mapping bug**.
Extending the shader to honor the full rotation basis (replace `rotZ`/`yaw` with a
real `transpose(R)` inverse, rotate the impostor box and normal by `R`, upgrade the
voxel AABB scatter) is out of scope here and listed in §8.

**Coordinate convention.** The renderer is **Z-up, ground = X-Y plane** (confirmed:
`ResizeSystem.ts:108` `lookAt(..., [0,0,1])`; `SunLight.ts` "Z-up frame"). Therefore
the engine adopts:

- **Up axis = +Z.** Ground is the X-Y plane at z = 0.
- **Gravity = `(0, 0, -9.81)`** (down the −Z axis).
- **Shape z-origin is the BOTTOM of the shape, not the center**
  (`sdf.shader.ts:20-21`: `transform col3.z = baseZ (bottom)`, the shader lifts to
  `center.z = baseZ + Height/2`). Rapier's body translation is the **centroid**, so the
  state→matrix sync must write `baseZ = physicsCenterZ − halfHeight` and set
  `Height = 2·halfHeight`, so the SDF center lands exactly on the physics center.

---

## 2. Architecture

### DI singletons — recommend the **leanest: two singletons** (`EngineDI` + `RenderDI`)

`unknown` uses three (`GameDI`, `RenderDI`, `PluginDI`). For a generic, no-game engine
we keep two and drop `PluginDI` (its only purpose is game extension hooks; the demo has
none — re-add later if needed). Both are **plain module-level mutable objects**, not
classes, filled by `createEngine`.

**`src/DI/EngineDI.ts`** — world + physics + frame entry point:
```ts
export const EngineDI: {
  width: number;
  height: number;
  world: EngineWorld;            // bitecs world ({components, time} context)
  physicalWorld: PhysicalWorld;  // Rapier World (re-export of @dimforge/rapier3d-simd World)
  tick: (delta: number) => void; // the deterministic frame
  destroy: () => void;
  setRenderTarget: (canvas: HTMLCanvasElement | null | undefined) => void;
} = {} as any;
```

**`src/DI/RenderDI.ts`** — GPU handles + the render closure (mirrors `unknown`'s
`RenderDI`, dropping the RC `lighting` field):
```ts
export const RenderDI: {
  enabled: boolean;
  canvas: HTMLCanvasElement;
  device: GPUDevice;
  context: GPUCanvasContext;
  renderFrame?: (delta: number) => void; // OPTIONAL — headless when absent
  destroy?: () => void;
} = { enabled: false } as any;
```
`renderFrame` is optional so the world can run **headless** (no canvas) — `tick` calls
`RenderDI.renderFrame?.(delta)`. This preserves `unknown`'s headless-training property
for free, even though the demo always attaches a canvas.

### The ECS world — **new `createEngineWorld`** that REUSES `createRenderComponents`

We cannot call renderer3d_2's `createWorld()` directly: it builds a world whose
`components` is exactly `RenderComponents`, with no room for our physics components. We
mirror `unknown`'s `createGameWorld`: build the bitecs world with a `{components, time}`
context, then **spread** renderer components and engine components together.

```ts
// src/ECS/createEngineWorld.ts
import { createWorld as createBitecsWorld } from "bitecs";
import { createRenderComponents, type RenderComponents } from "../../../renderer3d_2/src/ECS/world.ts";
import { createEngineComponents } from "./components.ts";

export function createEngineWorld(): EngineWorld {
  const context = {
    components: null as unknown as EngineComponents,
    time: { delta: 0, elapsed: 0, then: performance.now() },
  };
  const world = createBitecsWorld(context) as EngineWorld;
  context.components = {
    ...createRenderComponents(world),   // GlobalTransform, LocalTransform, Shape, Color, Height, LightEmitter, ...
    ...createEngineComponents(world),   // RigidBodyRef, RigidBodyState3D
  };
  return world;
}
export type EngineComponents = RenderComponents & ReturnType<typeof createEngineComponents>;
export function getEngineComponents(world): EngineComponents { /* throws if absent */ }
```

**Important nuance discovered in source:** in renderer3d_2 `GlobalTransform` /
`LocalTransform` are **module-level singletons** (not per-world factory products —
`Transform.ts:18-19`), and `createRenderComponents` returns those same singletons by
reference. renderer3d_2's own systems (`createDrawShapeSystem`, `createTransformSystem`,
`createVoxelSystem`) read components via `getRenderComponents(world)`, which reads
`world.components`. Because we spread `createRenderComponents(world)` into our
`world.components`, those systems find exactly the components they expect. **This is
why our world is drop-in compatible with every renderer3d_2 system** — they only ever
look at `world.components.{GlobalTransform,Shape,...}`, all of which we provide.

### Component set — reuse vs. new (justified per component)

| Component | Decision | Justification |
|---|---|---|
| `GlobalTransform`, `LocalTransform` | **Reuse** (renderer3d_2 `Transform.ts`) | The render pass reads `GlobalTransform.matrix.getBatch(eid)`. The sync system must write into the exact same column the renderer reads. `matrix` is `NestedArray.f32(16)` → a `mat4` view. Reusing is mandatory, not optional. |
| `Shape` | **Reuse** (`createShapeComponent`) | The SDF pass keys off `Shape.kind`/`Shape.values`. Box footprint, sphere radius live here. Set by the `create*` shape factories. |
| `Color` | **Reuse** | Read by the draw pass (`getArray(eid)` rgba) and by the light feed. |
| `Height` | **Reuse** | The vertical extent. Must equal `2·halfHeight` so SDF center = physics center (see §1, §4). Set by shape factories. |
| `LightEmitter` | **Reuse** | The VCT light feed discovers `[LightEmitter, LocalTransform]`. The demo uses only the sun, so emitters are optional — but reuse keeps the door open. |
| `Roundness`, `Thinness`, `Blurness`, `Translucency`, `Rope` | **Reuse (unused)** | They come free in the spread; the demo ignores them. |
| **`RigidBodyRef`** | **New** | Stores the Rapier body handle (`pid`) per `eid` and maintains the reverse `Map<pid, eid>`. Identical shape to `unknown`'s — Rapier 3D handles are also numeric. |
| **`RigidBodyState3D`** | **New** | The per-frame physics snapshot. **This is the single biggest 2D→3D delta:** `position` is `[x,y,z]`, `rotation` is a **quaternion** `[x,y,z,w]` (2D stored a scalar angle). |

We deliberately **do not** port `unknown`'s `Impulse`/`TorqueImpulse` command
components or `createApplyImpulseSystem`: the demo applies no forces (gravity does all
the work). The write-direction bridge is a documented future extension (§8), not part
of the minimum deliverable.

### The pid↔eid mapping

Two independent ID spaces joined only by `RigidBodyRef`:
- `RigidBodyRef.id[eid] = pid` (forward: entity → Rapier handle), and
- a module-level `Map<number, number>` `pid → eid` (reverse, for collision-event
  resolution), exposed as `getEntityIdByPhysicalId(pid)`.

Rapier handle **0** is reserved as the empty-memory sentinel (a `Float64Array` row
defaults to 0). The physics init creates one throwaway body so handle 0 is never a real
entity — mirrors `unknown`.

---

## 3. Physics layer

### Init + import pattern (Vite wasm) — **no `RAPIER.init()`**

`@dimforge/rapier3d-simd` is the wasm-**bundler** flavor (same as `unknown`'s
2d-simd). Import named exports directly and synchronously; `vite-plugin-wasm` +
`vite-plugin-top-level-await` make the top-level wasm instantiation transparent. The
skeleton's `config.vite.ts` already excludes the package from `optimizeDeps` — correct,
leave it.

```ts
// src/Physics/initPhysicalWorld.ts
import { World, Vector3 } from "@dimforge/rapier3d-simd";
// Optional raw params subpath — confirm filename after install (§8):
//   import { RawIntegrationParameters } from "@dimforge/rapier3d-simd/rapier_wasm3d";

export type PhysicalWorld = World;

export function initPhysicalWorld(): PhysicalWorld {
  const gravity = new Vector3(0, 0, -9.81); // Z-up world → gravity down −Z
  const world = new World(gravity);
  world.integrationParameters.lengthUnit = 100; // match unknown's world scale; optional
  world.integrationParameters.numSolverIterations = 4;
  reserveHandleZero(world); // dummy body so real handles start at 1
  return world;
}
```
(If `world.integrationParameters` mutation is not exposed, fall back to constructing
`new World(gravity, new RawIntegrationParameters())` with fields set — the 2D code does
exactly this. Default `new World(gravity)` is sufficient for the demo; tune later.)

### Body + collider factories

`src/Physics/createBody.ts` — generic rigid-body desc builder:
```ts
import { RigidBodyDesc, Vector3 } from "@dimforge/rapier3d-simd";
// type: "dynamic" | "fixed"; rot is a quaternion {x,y,z,w}, identity = {0,0,0,1}
export function createBody(world, { type, x, y, z, rot, ... }): RigidBody {
  const desc = (type === "fixed" ? RigidBodyDesc.fixed() : RigidBodyDesc.dynamic())
    .setTranslation(x, y, z)                 // 3 args (2D took 2)
    .setRotation(rot ?? { x: 0, y: 0, z: 0, w: 1 }); // quaternion (2D took scalar rad)
  return world.createRigidBody(desc);
}
```
`src/Physics/createRigid.ts` — collider + handle return:
```ts
import { ColliderDesc } from "@dimforge/rapier3d-simd";
export function createRigidBox(world, body, hx, hy, hz): number {
  const desc = ColliderDesc.cuboid(hx, hy, hz).setDensity(1); // 3 half-extents
  world.createCollider(desc, body);
  return body.handle;
}
export function createRigidBall(world, body, r): number {
  const desc = ColliderDesc.ball(r).setDensity(1);            // identical to 2D
  world.createCollider(desc, body);
  return body.handle;
}
// Ground: a fixed cuboid that is wide+long and thin in Z, OR a half-space.
```
Collision-group packing `(belongs << 16) | interacts`, `.setSensor/.setActiveEvents`
chaining etc. are **API-identical to 2D** — copy verbatim if/when needed; the demo
needs none.

### Step

`world.step()` (no event queue needed for the demo — nothing consumes contacts). The
`EventQueue` from `@dimforge/rapier3d-simd` and `world.step(eventQueue)` are the path to
add when collision callbacks are wired (§8).

---

## 4. The physics → render sync

The seam is two systems plus the renderer's transform + render passes, in a strict
order. The **read direction** is the only flow in the demo (Rapier → state → matrix);
there is no write-back this milestone.

### `src/ECS/Systems/createRigidBodyStateSystem.ts` — Rapier → `RigidBodyState3D`
```ts
const entities = query(world, [RigidBodyRef, RigidBodyState3D]);
for (let i = 0; i < entities.length; i++) {
  const eid = entities[i];
  const pid = RigidBodyRef.id[eid];
  const body = physicalWorld.getRigidBody(pid);
  if (body.isSleeping()) continue;          // sleeping = unchanged since last sync
  const p = body.translation();             // {x,y,z}  — physics CENTER
  const q = body.rotation();                // {x,y,z,w} quaternion (2D returned scalar)
  const lv = body.linvel();                 // {x,y,z}
  const av = body.angvel();                 // {x,y,z}
  RigidBodyState3D.update(eid, p.x,p.y,p.z, q.x,q.y,q.z,q.w, lv.x,lv.y,lv.z, av.x,av.y,av.z);
}
```
Note: with the **object-style** body API (`body.translation()` returns a plain
`{x,y,z}`), there are **no WASM handles to `.free()`** — unlike `unknown`'s raw-bodies
path (`rawBodies.rbTranslation(pid)` which returns a freeable WASM object). We use the
high-level `RigidBody` accessors, which return GC'd JS objects. (If profiling later
shows allocation pressure, switch to the raw API and `.free()`; not needed for the demo.)

### `src/ECS/Systems/createApplyRigidBodyToTransformSystem.ts` — state → `LocalTransform.matrix`
This is the load-bearing matrix write. We write into **LocalTransform** (not
GlobalTransform directly) so the renderer's `createTransformSystem` then copies
local→global uniformly for every entity — keeping a single, consistent path. We use
`mat4.fromRotationTranslation` (full 3D rotation, even though the shader only honors its
yaw — see §1) and apply the **bottom-origin Z offset**:
```ts
import { mat4, quat, vec3 } from "gl-matrix";
const _q = quat.create(), _t = vec3.create();
const entities = query(world, [LocalTransform, RigidBodyRef, RigidBodyState3D, Height]);
for (let i = 0; i < entities.length; i++) {
  const eid = entities[i];
  const hz = Height.value[eid] * 0.5;                       // half vertical extent
  quat.set(_q, RBS.rotation.get(eid,0), RBS.rotation.get(eid,1),
               RBS.rotation.get(eid,2), RBS.rotation.get(eid,3));
  vec3.set(_t, RBS.position.get(eid,0), RBS.position.get(eid,1),
               RBS.position.get(eid,2) - hz);               // baseZ = centerZ − hz
  const m = LocalTransform.matrix.getBatch(eid);            // mat4 view (16 f32)
  mat4.fromRotationTranslation(m, _q, _t);                  // writes full 4x4 in place
}
```
Because we set `Height = 2·hz` at spawn, the shader recovers
`center.z = baseZ + Height/2 = (centerZ − hz) + hz = centerZ` — exact alignment with
the physics centroid. (Cheap yaw-only alternative for the common case:
`setMatrixTranslate(m, x, y, z−hz)` + `setMatrixRotateZ(m, yaw)` — we prefer the full
`fromRotationTranslation` since it costs the same and is future-proof for a shader
upgrade.)

### EXACT frame order (the spine)

`createEngine`'s `tick(delta)` does, in order:
```
1. physicalFrame(delta):
   a. execTransformSystem()              // (renderer3d_2 TransformSystem) local→global compose
                                         //    — runs first so any static/child offsets are set;
                                         //      cheap and harmless for flat scenes
   b. physicalWorld.step()               // Rapier integrates (gravity)
   c. syncRigidBodyState()               // Rapier body → RigidBodyState3D   (READ)
   d. applyRigidBodyToLocalTransform()   // RigidBodyState3D → LocalTransform.matrix (mat4)
2. RenderDI.renderFrame?.(delta)         // the render block below
```

The **render block** (`renderFrame`, built in `createRenderTarget`) is the renderer3d_2
sequence, function-for-function (from the render-api finding), with our transform pass
folded in so global matrices are current:
```
resizeSystem()                                   // ResizeSystem: refresh viewProjMatrix/cameraRayDir
if (canvas resized) { rebuild frame textures + frameTick; voxel.recreate(depth,normal,render,emission) }
execTransformSystem()                            // LocalTransform → GlobalTransform (the matrices the draw reads)
// (optional) updateLights() → voxel.setLights(flat, n, colorsFlat)   // demo: sun only, skip or no-op
shapeSystem.prepare()                            // packs GlobalTransform/Shape/Color → GPU + CPU arrays
const encoder = device.createCommandEncoder();
frameTick(encoder, delta);                       // main SDF G-buffer pass → calls shapeSystem.drawShapes(passEncoder)
if (SunLight.enabled) voxel.sunDepth(encoder);   // shadow map from sun POV
voxel.voxelize(encoder);                         // scatter into radiance volume
voxel.mips(encoder);                             // mip pyramid
voxel.probe(encoder);                            // SH-L1 irradiance probes
voxel.cone(encoder);                             // VCT gather (half-res)
voxel.composite(encoder);                        // final lit HDR image
present(encoder, voxel.compositeOutputTexture);  // blit to swapchain
device.queue.submit([encoder.finish()]);
```

Hard ordering constraints (from the render-api finding, must not be reordered):
- `resizeSystem()` first, so `viewProjMatrix` is current **before** `prepare()`.
- `execTransformSystem()` → `shapeSystem.prepare()` (prepare reads `GlobalTransform`).
- `prepare()` before any encoder pass (it uploads uniforms + fills the CPU arrays the
  voxel pass reads).
- `sunDepth` before `voxelize`/`composite`; then `voxelize → mips → probe → cone →
  composite`; `setLights` (if used) before `cone`.
- The G-buffer textures handed to `createVoxelSystem` must be the **same objects**
  passed to `createFrameTick`, and re-fed via `voxel.recreate` on resize.

Note `execTransformSystem` runs **twice** (once in `physicalFrame` for hierarchy
compose, once in the render block). For the flat demo scene the first call is
effectively redundant; keeping it mirrors `unknown`'s order and is correct for future
parent/child bodies. (Optimization: collapse to one call — deferred.)

---

## 5. Entity factory recipe

Generic "give an entity a body + a render shape", in `src/ECS/Entities/RigidShapes.ts`.
Order: create the render entity (gets Transform/Shape/Color/Height), create the physics
body+collider (gets handle), then attach the bridge + state components and register the
mapping. Returns `[eid, pid]`.

```ts
import { createRectangle, createSphere } from "../../../renderer3d_2/src/ECS/Entities/Shapes.ts";

// Box: render = a rectangle footprint extruded by `depth`; physics = cuboid(hx,hy,hz).
export function createRigidBox(world, physicalWorld, { x, y, z, sx, sy, sz, color, density? }) {
  const { RigidBodyRef, RigidBodyState3D, Height } = getEngineComponents(world);
  const hx = sx/2, hy = sy/2, hz = sz/2;
  // render: footprint = sx×sy in XY, extruded sz along Z; z is the BOTTOM.
  const eid = createRectangle(world, { x, y, z: z - hz, width: sx, height: sy, color, depth: sz });
  // (createRectangle sets Height = sz internally via `depth`; if not, Height.set$(eid, sz).)
  const body = createBody(physicalWorld, { type: "dynamic", x, y, z, rot: IDENTITY_QUAT });
  const pid  = createRigidBox_collider(physicalWorld, body, hx, hy, hz); // returns body.handle
  RigidBodyRef.addComponent(world, eid, pid);        // bridge + register pid→eid
  RigidBodyState3D.addComponent(world, eid);         // zeroed snapshot
  return [eid, pid];
}

// Sphere: render = createSphere (sets Height = 2r); physics = ball(r). Rotation-invariant.
export function createRigidSphere(world, physicalWorld, { x, y, z, radius, color }) {
  const { RigidBodyRef, RigidBodyState3D } = getEngineComponents(world);
  const eid = createSphere(world, { x: x, y: y, z: z - radius, radius, color }); // z = bottom = center−r
  const body = createBody(physicalWorld, { type: "dynamic", x, y, z, rot: IDENTITY_QUAT });
  const pid  = createRigidBall(physicalWorld, body, radius);
  RigidBodyRef.addComponent(world, eid, pid);
  RigidBodyState3D.addComponent(world, eid);
  return [eid, pid];
}

// Ground: a FIXED, thin, wide box. Render = a large flat rectangle; physics = thin cuboid.
export function createGround(world, physicalWorld, { size = 200, thickness = 1, z = 0, color }) {
  const { RigidBodyRef, RigidBodyState3D } = getEngineComponents(world);
  const hz = thickness/2;
  const eid = createRectangle(world, { x:0, y:0, z: z - thickness, width: size, height: size, color, depth: thickness });
  const body = createBody(physicalWorld, { type: "fixed", x:0, y:0, z: z - hz, rot: IDENTITY_QUAT });
  const pid  = createRigidBox_collider(physicalWorld, body, size/2, size/2, hz);
  RigidBodyRef.addComponent(world, eid, pid);
  RigidBodyState3D.addComponent(world, eid); // fixed bodies never wake, so this stays zero-synced; harmless
  return [eid, pid];
}
```

**Exact `Shapes.ts` signatures used** (from the render-api finding):
`createSphere(world, {x,y,z,radius,color})` (sets `Height = 2r`),
`createRectangle(world, {x,y,z,width,height,color,roundness?,depth?})` where `z` =
baseZ (bottom) and `depth` = vertical extrusion → drives `Height`. `color` is
`[r,g,b,a]`. The factory passes `z = center − halfHeight` so the **initial** render
matches the physics centroid before the first sync (the sync then maintains it every
frame).

The crucial invariant the factory establishes: **`Height` (render) = the body's full Z
extent**, so the §4 sync's `baseZ = centerZ − Height/2` offset is exact for every kind.

---

## 6. Demo scene

`src/demo.ts` — concrete scene + main loop. No GUI/Stats required (keep it minimal; a
HUD line already exists in `index.html`).

**Scene:**
- **Ground:** `createGround(world, pw, { size: 80, thickness: 1, z: 0, color: [0.18,0.2,0.24,1] })`.
- **Falling boxes:** ~6 dynamic boxes, `sx=sy=sz=2`, spawned in a loose grid at
  `z ∈ [6 … 16]` with small XY jitter, varied colors. They fall and stack on the ground.
- **Falling spheres:** ~5 dynamic spheres, `radius=1`, spawned at `z ∈ [8 … 20]` over the
  ground; they bounce/settle (rotation-invariant, always render correctly).
- **Sun (VCT):** `SunLight.enabled = true; SunLight.angle = 2.4; SunLight.elevation =
  0.95; SunLight.intensity = 0.9; SunLight.color = [1.0, 0.93, 0.82]` (the demo's tuned
  warm values).
- **Camera:** `setCameraPosition(0, 0); cameraZoom.value = 14; cameraElevation.value =
  60; cameraAzimuth.value = 45`. Optionally orbit by advancing `cameraAzimuth.value`
  each frame for a slow turntable (`setCameraAzimuth(deg)`), which makes the 3D depth
  read clearly.

**Setup (once):**
```ts
const canvas = document.getElementById("c") as HTMLCanvasElement;
const engine = await createEngine({ canvas });   // builds world, physics, systems, render target
buildDemoScene(engine.world, engine.physicalWorld);   // the spawns above + SunLight + camera
```

**Main loop body:**
```ts
let then = performance.now();
function loop(now: number) {
  const delta = Math.min(now - then, 16.6667) / 1000; // clamp, seconds (Rapier dt)
  then = now;
  // optional turntable: setCameraAzimuth(cameraAzimuth.value + delta * 10);
  engine.tick(delta);          // physicalFrame → renderFrame (the full §4 spine)
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
```
`engine.tick` internally runs `physicalWorld.step()` (gravity drops everything), the two
sync systems, then the renderer3d_2 pass chain ending in `present(...)`.

---

## 7. File-by-file plan (ordered by dependency)

All under `packages/engine/src/`. Renderer imports are relative
(`../../renderer3d_2/src/...`). Build bottom-up.

1. **`Physics/initPhysicalWorld.ts`** — `initPhysicalWorld(): PhysicalWorld`,
   `export type PhysicalWorld`, `reserveHandleZero(world)`. Gravity `(0,0,-9.81)`,
   lengthUnit/solver tuning. *(deps: rapier3d-simd)*
2. **`Physics/createBody.ts`** — `createBody(world, {type,x,y,z,rot}): RigidBody`,
   `IDENTITY_QUAT = {x:0,y:0,z:0,w:1}`. `RigidBodyDesc.dynamic()/.fixed()`,
   `setTranslation(x,y,z)`, `setRotation(quat)`. *(deps: rapier3d-simd)*
3. **`Physics/createRigid.ts`** — `createRigidBox(world, body, hx,hy,hz): number`,
   `createRigidBall(world, body, r): number`, ground helper. Returns `body.handle`.
   `ColliderDesc.cuboid(hx,hy,hz)` / `.ball(r)`. *(deps: rapier3d-simd)*
4. **`ECS/Components/RigidBodyRef.ts`** — `createRigidBodyRefComponent =
   defineComponent((ref) => {...})`: `id = TypedArray.f64(size)`, `addComponent(world,
   eid, pid)`, `clear(eid)`, `dispose()`; module `Map<pid,eid>` +
   `getEntityIdByPhysicalId(pid): number`. *(deps: renderer3d_2 ECS/utils
   `defineComponent`, utils `TypedArray`)*
5. **`ECS/Components/RigidBodyState3D.ts`** — `createRigidBodyState3DComponent =
   defineComponent((ref) => {...})`: `position = NestedArray.f64(3,size)`, `rotation =
   NestedArray.f64(4,size)` (quat), `linvel = NestedArray.f64(3,size)`, `angvel =
   NestedArray.f64(3,size)`; `addComponent(world,eid)` (zero-fills, identity quat at
   rotation[3]=1), `update(eid, px,py,pz, qx,qy,qz,qw, lx,ly,lz, ax,ay,az)`. *(deps:
   renderer3d_2 ECS/utils, utils `NestedArray`)*
6. **`ECS/components.ts`** — `createEngineComponents(world)` returns
   `{ RigidBodyRef: createRigidBodyRefComponent(world), RigidBodyState3D:
   createRigidBodyState3DComponent(world) }`. *(deps: 4, 5)*
7. **`ECS/createEngineWorld.ts`** — `createEngineWorld(): EngineWorld`,
   `getEngineComponents(world)`, `type EngineComponents`, `type EngineWorld`. Spreads
   `createRenderComponents(world)` + `createEngineComponents(world)` into
   `{components, time}` context. *(deps: 6, renderer3d_2 `ECS/world.ts`)*
8. **`ECS/Systems/createRigidBodyStateSystem.ts`** —
   `createRigidBodyStateSystem(world, physicalWorld): () => void`. Query
   `[RigidBodyRef, RigidBodyState3D]`, skip sleeping, read
   `translation/rotation/linvel/angvel`, `RigidBodyState3D.update(...)`. *(deps: 5, 7,
   rapier3d-simd)*
9. **`ECS/Systems/createApplyRigidBodyToTransformSystem.ts`** —
   `createApplyRigidBodyToTransformSystem(world): () => void`. Query `[LocalTransform,
   RigidBodyRef, RigidBodyState3D, Height]`, `mat4.fromRotationTranslation` with
   bottom-Z offset → `LocalTransform.matrix.getBatch(eid)`. *(deps: 5, 7, renderer3d_2
   `Transform.ts`, gl-matrix)*
10. **`ECS/Entities/RigidShapes.ts`** — `createRigidBox`, `createRigidSphere`,
    `createGround`. Composes render `create*` + physics body/collider +
    `RigidBodyRef.addComponent` + `RigidBodyState3D.addComponent`; returns `[eid, pid]`.
    *(deps: 2, 3, 7, renderer3d_2 `Entities/Shapes.ts`)*
11. **`DI/RenderDI.ts`** — the `RenderDI` singleton (§2). *(no deps)*
12. **`DI/EngineDI.ts`** — the `EngineDI` singleton (§2). *(deps: 7 for types)*
13. **`createRenderTarget.ts`** — `createRenderTarget(world, canvas): { renderFrame,
    destroy }`. Does the render-api setup: `initWebGPU`, `createFrameTextures`,
    `createFrameTick(..., ({passEncoder}) => shapeSystem.drawShapes(passEncoder))`,
    `createDrawShapeSystem`, `createVoxelSystem({device,canvas,sceneInstances,
    depthTexture,normalTexture,albedoTexture:renderTexture,emissionTexture})`,
    `createPresent`, `createResizeSystem`, `createTransformSystem(world, stubChildren)`.
    Returns the per-frame `renderFrame(delta)` closure implementing the exact §4 render
    block (incl. resize rebuild + `voxel.recreate`). Fills `RenderDI`. *(deps: 11,
    many renderer3d_2 modules)*
14. **`createEngine.ts`** — `createEngine({canvas, width?, height?}): Promise<EngineDI>`.
    Builds the world (7), physics world (1), instantiates the two sync systems (8, 9)
    and `createTransformSystem` (renderer3d_2), assembles `tick(delta)` =
    `physicalFrame` + `RenderDI.renderFrame?.(delta)`, wires `setRenderTarget` (calls
    13), fills `EngineDI`. *(deps: 1, 7, 8, 9, 12, 13, renderer3d_2 `TransformSystem`)*
15. **`demo.ts`** — entry (referenced by `index.html`). `createEngine`, build scene
    (10) + sun + camera, run the rAF loop. *(deps: 14, 10, renderer3d_2 `SunLight`,
    `ResizeSystem` camera setters)*

Supporting constants (`IDENTITY_QUAT`, `stubChildren`) can live inline or in a tiny
`src/lib/constants.ts`.

---

## 8. Risks & open questions

1. **Rapier 3D exact API surface.** The findings infer the 3D API from the 2D code +
   convention. Verify after `npm install` against the actual `.d.ts`:
   - `body.rotation()` returns `{x,y,z,w}`, `body.translation()`/`linvel()`/`angvel()`
     return `{x,y,z}` — **confirm** (vs. needing `.free()` raw-API objects).
   - `setRotation` accepts a plain `{x,y,z,w}` object (vs. a `Quaternion` instance).
   - `setAngvel` arg type (likely `Vector3`/`{x,y,z}`); not used by the demo.
   - `world.integrationParameters` mutability vs. passing `RawIntegrationParameters` to
     the `World` ctor — and the **raw-params subpath filename**
     (`@dimforge/rapier3d-simd/rapier_wasm3d`, by 2D convention; `ls
     node_modules/@dimforge/rapier3d-simd | grep rapier_wasm` to confirm). The demo can
     ship with `new World(gravity)` and skip raw params entirely.
   - `body.isSleeping()` name (vs. `unknown`'s raw `rbIsSleeping`).
2. **Full 3D rotation does not render (the central renderer limitation).** Tumbling
   boxes render upright; only yaw + translation + spheres are faithful (§1). For the
   demo this is acceptable. If faithful tumbling is later required, the shader work is
   substantial: in `sdf.shader.ts`/`sceneSDF.wgsl.ts` replace the `yaw`/`rotZ`
   reconstruction with a real `transpose(R)` inverse for ray + origin, rotate the
   impostor box and the surface normal by the full `R`, and upgrade the voxel AABB
   scatter (`createDrawShapeSystem.ts` `cpuTransform` consumers) — flagged, out of scope.
3. **Reverse-Z / projection.** The engine never builds a projection; it only writes
   `LocalTransform` matrices and lets `ResizeSystem`/`viewProjMatrix` (reverse-Z,
   ortho) do its thing. No action needed — listed only so it isn't re-derived.
4. **Two systems named "renderer" vs "renderer3d_2".** The engine imports **only**
   `renderer3d_2`. Ensure no accidental `../../renderer/...` imports leak in (the old 2D
   `renderer` has the Z-only transform helpers and a different world). All paths in §7
   are `../../renderer3d_2/src/...`.
5. **npm install / workspaces.** `@dimforge/rapier3d-simd` is in `package.json` but not
   yet in `node_modules` (only 2d/2d-simd are present). Run install at the repo root so
   the monorepo workspace resolves it; confirm the wasm-bundler dist
   (`rapier_wasm3d.js` doing `import * as wasm from "./rapier_wasm3d_bg.wasm"`) is what
   landed (the `^0.19.3` range may resolve to an older dist, as it did for 2D). If Vite
   chokes on the wasm import, double-check the package is in `optimizeDeps.exclude`
   (it already is).
6. **`createRenderComponents` returns module-singleton transforms.** `GlobalTransform`/
   `LocalTransform` are shared module singletons, so two engine worlds in one page would
   alias the same transform columns. The demo runs a single world — fine. Multi-world
   in one tab is a known renderer3d_2 limitation, not introduced by the engine.
7. **`shapeSystem.prepare()` clamps to `MAX_INSTANCE_COUNT`.** The demo's ~12 bodies are
   far under any cap; noted so scaling the scene later checks this.
8. **`delta` units.** Rapier integrates with its own fixed `dt` from
   `integrationParameters` unless `world.timestep` is set; `world.step()` ignores the
   JS `delta` we pass to the renderer. For deterministic settling this is fine; if
   frame-rate-coupled physics is wanted, set `world.timestep = delta` before `step()`.
   The renderer's `delta` (clamped to 16.6667 ms) is for animation timing only.
