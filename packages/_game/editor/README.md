# viewer

An entity viewer over the `engine` package: pick a procedurally-built entity from the
selector and inspect it in the 3D viewport. Entities are defined in code (no scene
editing) under `src/Entities/`.

## Run

```bash
npm run dev   # in this package; serves on port 3355 under COOP/COEP (SharedArrayBuffer)
```

## Controls

- **Entity selector** (left panel) — choose which entity to display. The choice is
  remembered across reloads via `localStorage` (`viewer.entity`).
- **Regenerate** — rebuild the current entity (procedural builders use randomness).
- **Drag** the viewport — orbit the camera. **Wheel** — zoom.

## Adding an entity

1. Write a builder `build(world: EngineWorld): number` that returns the **root** eid.
   Create a root via `adoptEntity(world, createEntityId(world))` + `addTransformComponents`
   + `Children.addComponent`, then build the parts as children (`Children.addChild(root, part)`)
   in coords LOCAL to the root — the transform system composes them. The viewer holds only
   the root and clears the whole subtree (`removeEntityTree`). For static decor no physics is needed.
2. Register it in `src/Entities/registry.ts`.

Hierarchy is real transform parenting: parts are children of the root, so the whole entity
moves/clears as one. (One level deep for now; see `engine` `Children` + `TransformSystem`.)

Current entities: **tree** (`src/Entities/tree.ts`) — a root pivot with a trunk box + a
cluster of canopy spheres as children, randomized per build.
