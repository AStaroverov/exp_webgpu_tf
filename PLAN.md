# Plan: Obstacle Spatial Map + 128 Rays + Remove History

## Summary
Four changes:
1. Add edge walls to battlefield (all scenarios get obstacles from day 1)
2. Remove history system (T=4 → T=1)
3. Increase rays 32 → 128
4. Add 16×16 obstacle grid as new input

---

## Phase 0: Edge Walls on All Maps

**Goal**: Every scenario has obstacles from the first training step, so the network never learns to ignore the obstacle grid input.

**Problem**: Scenario 0 (`createScenario1v1Random`) has zero obstacles. If the grid is always empty early in training, the network zeros out the grid pathway and never recovers.

**Solution**: Add physical walls along the 4 edges of the battlefield in `createBattlefield()`. Since ALL scenarios go through `createBattlefield()` → `createScenarioCore()`, every episode will have edge walls.

### 0.1 New utility — `packages/ml-common/Curriculum/Utils/addEdgeWalls.ts`

```typescript
export function addEdgeWalls(width: number, height: number): void
```

Creates 4 rectangular wall segments:
- **Top wall**: `x = width/2, y = 0`, size `width × WALL_THICKNESS`
- **Bottom wall**: `x = width/2, y = height`, size `width × WALL_THICKNESS`
- **Left wall**: `x = 0, y = height/2`, size `WALL_THICKNESS × height`
- **Right wall**: `x = width, y = height/2`, size `WALL_THICKNESS × height`

`WALL_THICKNESS ≈ 30` — enough to be visible on the 16×16 grid (cell size ~37-87 at field 600-1400).

**Implementation**: Use `createBuildingParts()` directly with handcrafted `BuildingPartData[]` (type `'wall'`). This gives each segment the `Obstacle` + `Hitable` components automatically. Alternatively, use `createRectangleRR()` + manually add `Obstacle.addComponent()`.

Simplest approach: 4 calls to `createBuilding()` with parameters that produce solid walls:
```typescript
// Example: top wall
createBuilding({
    x: width / 2,
    y: 0,
    cols: Math.ceil(width / 30),  // enough cols to span the width
    rows: 1,
    cellSize: 30,
    wallThickness: 30,
    destructionThreshold: 1,      // no destruction → fully solid
    interiorWallChance: 0,
    noiseScale: 0,
});
```

### 0.2 Call site — `packages/ml-common/Curriculum/createBattlefield.ts`

Add after `createGame()`:
```typescript
const game = createGame({ width: size, height: size });
addEdgeWalls(size, size);  // NEW — edge obstacles on every map
```

This ensures:
- Scenario 0 (1v1 random) → has edge walls
- Scenario 1 (diagonal) → has edge walls + center building
- All other scenarios → edge walls + their own obstacles
- Grid is never fully empty

---

## Phase 1: Remove History

**Goal**: T=1, no temporal frames. Simplifies pipeline, reduces token count.

### 1.1 `packages/ml-common/historyConfig.ts`
- `HISTORY_OFFSETS = [0]`
- `HISTORY_LENGTH = 1`

### 1.2 `packages/ml-common/InputArrays.ts`
- `StateHistory` type stays (still `InputArrays[]`, length 1)
- `assembleStateHistory` / `assembleCurrentStateHistory` — still work with length 1, no code changes needed
- Can remove the `// [t, t-3, t-6, t-9, t-12]` comment

### 1.3 `packages/ml-common/InputTensors.ts`
- No code changes needed (T=1 works naturally)
- Token counts drop: rays `T*32=128 → 1*128=128`, enemies `T*5=20 → 5`, etc.

### 1.4 `packages/ml/src/Models/Inputs.ts`
- `TemporalPositionLayer` becomes no-op with T=1, can remove calls to `addTemporalEncoding`
- Simplify shapes in comments

### 1.5 `packages/ml/src/Models/Networks/v10.ts`
- No structural changes needed (perceiver adapts to fewer tokens)

### 1.6 `packages/tanks/src/Plugins/Pilots/Agents/NetworkModelManager.ts`
- History assembly still works — just assembles 1 frame now

**Token count impact** (before → after):
- Tank: 4 → 1
- Turret: 4 → 1
- Rays: 128 → 32 (before ray increase)
- Enemies: 20 → 5
- Allies: 20 → 5
- Bullets: 32 → 8
- **Total: ~228 → ~52 tokens**

---

## Phase 2: Increase Rays 32 → 128

**Goal**: 4× denser ray coverage for better local awareness.

### 2.1 `packages/tanks/src/Plugins/Pilots/Components/TankState.ts`
```
RAYS_COUNT = 32 → 128
```
- `raysData` NestedArray size: `RAY_BUFFER * 128 = 896` per entity (was 224)
- Everything else uses `RAYS_COUNT` constant — cascades automatically

### 2.2 Files that auto-adjust via constants (no code changes):
- `InputArrays.ts` — uses `RAY_SLOTS`, `RAY_FEATURES_DIM`
- `InputTensors.ts` — uses `RAY_SLOTS`
- `Inputs.ts` — uses `RAY_SLOTS`, `RAY_FEATURES_DIM`
- `Create.ts` — `RAY_SLOTS = RAYS_COUNT`

### 2.3 `snapshotTankInputTensor.ts`
- `UNIFIED_ANGLE_STEP` auto-adjusts: `2π/128 ≈ 2.8°` instead of `2π/32 ≈ 11.25°`
- Raycasting loop runs 128 iterations instead of 32
- `targetRayAngles` array auto-resizes

### 2.4 `packages/ml/src/Models/Networks/v10.ts`
- Consider increasing `summarizedRays` latent tokens: `4 → 8` (more rays → more latent capacity)

**Token count after Phase 1+2**: rays go from 32 to 128. Total: ~148 tokens

---

## Phase 3: Obstacle Spatial Map (16×16, 1 channel)

**Goal**: Give agent a "minimap" of obstacle layout across entire battlefield.

### Design
- 16×16 grid = 256 cells
- 1 channel: **obstacle presence** (binary: 0 or 1)
- Absolute coordinates (grid fixed to battlefield bounds)
- **Static** — computed once per episode, reused every frame
- Each cell token: 3 features — `[obstacle, cell_norm_x, cell_norm_y]`
  - `cell_norm_x = (col + 0.5) / GRID_SIZE - 0.5` → [-0.47 .. +0.47]
  - `cell_norm_y = (row + 0.5) / GRID_SIZE - 0.5`
  - Position features let the model know WHERE each cell is

### 3.1 Constants — `packages/ml/src/Models/Create.ts`
```typescript
export const GRID_SIZE = 16;
export const GRID_CHANNELS = 1;  // obstacle only
export const GRID_CELL_FEATURES = 3;  // obstacle + cell_x + cell_y
export const GRID_CELLS = GRID_SIZE * GRID_SIZE;  // 256
```

### 3.2 Obstacle grid computation — NEW `packages/ml-common/computeObstacleGrid.ts`

```typescript
export function computeObstacleGrid(
    world: World,
    width: number,
    height: number,
): Float32Array
```

**Algorithm**:
1. Query all entities with `[Obstacle, Hitable]` — this gives solid obstacles (walls, debris, rocks) and excludes floors
2. For each obstacle entity:
   - Get position from `GlobalTransform.matrix` (indices [12], [13] = x, y)
   - Get size from `Shape.values` (index 0 = width, 1 = height for rectangles)
   - Get rotation from `RigidBodyState.rotation`
3. Compute AABB of the rotated rectangle (4 corners → min/max x,y)
4. Mark all grid cells that the AABB overlaps as `1`
5. Return `Float32Array(GRID_SIZE * GRID_SIZE)` — flat binary grid

**Cell mapping**:
- `col = floor(x / cellWidth)`, `row = floor(y / cellHeight)`
- `cellWidth = width / GRID_SIZE`, `cellHeight = height / GRID_SIZE`
- Obstacle coordinates use view space (already offset-subtracted in GlobalTransform)

**Note**: AABB over-approximation is fine — minor inaccuracy for rotated thin walls, but grid resolution is already coarse.

### 3.3 InputArrays — `packages/ml-common/InputArrays.ts`

Add to `InputArrays` type:
```typescript
obstacleGrid: Float32Array,  // GRID_CELLS = 256
```

Modify `prepareInputArrays()` signature:
```typescript
export function prepareInputArrays(
    tankEid: number,
    width: number,
    height: number,
    obstacleGrid: Float32Array,  // NEW — pre-computed, shared across agents
): InputArrays
```
- Just copy reference (same grid for all agents): `obstacleGrid` assigned directly to result

Update `prepareRandomInputArrays()`:
- Add `obstacleGrid: new Float32Array(GRID_CELLS).map(() => randomRangeInt(0, 1))`

Update `checkInputArrays()`:
- Include `obstacleGrid` in validation

### 3.4 InputTensors — `packages/ml-common/InputTensors.ts`

New packing function `packGridF32`:
- Since grid is NOT multiplied by T (static per episode), pack differently from other fields
- Shape: `[B, GRID_CELLS, GRID_CELL_FEATURES]` = `[B, 256, 3]`
- For each batch item, generate features: for each cell `(row, col)`:
  - `obstacle = grid[row * GRID_SIZE + col]`
  - `cell_x = (col + 0.5) / GRID_SIZE - 0.5`
  - `cell_y = (row + 0.5) / GRID_SIZE - 0.5`

Add to `createInputTensors()` return array:
```typescript
tf.tensor3d(packGridF32(histories), [B, GRID_CELLS, GRID_CELL_FEATURES])
```

### 3.5 Model Inputs — `packages/ml/src/Models/Inputs.ts`

Add input:
```typescript
const obstacleGridInput = tf.input({
    name: name + '_obstacleGridInput',
    shape: [GRID_CELLS, GRID_CELL_FEATURES]  // [256, 3]
});
```

Add to `createInputs()` return and `convertInputsToTokens()`:
```typescript
const gridTok = toToken('grid', obstacleGridInput);  // [B, 256, dModel]
```
No temporal encoding needed (static data).

### 3.6 Network — `packages/ml/src/Models/Networks/v10.ts`

Add perceiver for grid:
```typescript
const summarizedGrid = summarize({
    name: modelName + '_summarizedGrid',
    heads: config.heads,
    length: 4,  // 4 latent tokens
    token: tokens.gridTok,
    perceiverDepth: ceil(2 * config.depth),
    transformerDepth: ceil(1 * config.depth),
});
```

Include in `getHeadsToken`:
```typescript
tf.layers.concatenate(...).apply([
    tokens.tankTok,
    tokens.turretTok,
    summarizedVehicle,
    summarizedRays,
    summarizedProjectiles,
    summarizedGrid,        // NEW
])
```

### 3.7 Grid computation call site

Where to call `computeObstacleGrid()`:
- In the training loop / episode setup, after battlefield is created
- Pass grid through to `snapshotTankInputTensor` → `prepareInputArrays`
- Cache per episode (obstacles don't change)

Need to find the exact call site — likely in:
- `packages/tanks/src/Plugins/Pilots/Agents/NetworkModelManager.ts` or
- `packages/ml-common/Curriculum/` episode setup

---

## Final Token Count (all phases)

| Input | Tokens |
|-------|--------|
| Tank | 1 |
| Turret | 1 |
| Rays | 128 |
| Enemies | 5 |
| Allies | 5 |
| Bullets | 8 |
| **Grid** | **256** |
| **Total** | **404** |

Latent tokens after perceivers:
- Vehicles: 6
- Rays: 8
- Bullets: 2
- Grid: 16
- Tank + Turret: 2
- **Total to final perceiver: 22 tokens**

---

## Implementation Order

1. **Phase 0** — Edge walls (ensures grid is never empty from training start)
2. **Phase 1** — Remove history (simplifies pipeline, reduces tokens)
3. **Phase 2** — 128 rays (one constant change + latent count tweak)
4. **Phase 3** — Obstacle grid (new feature, most code)
5. **Test** — Verify model builds, dimensions match, inference runs
6. **Train** — Compare with baseline
