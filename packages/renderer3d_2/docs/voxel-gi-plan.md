# Voxel GI — план (renderer3d_2)

Замена прошлого подхода (Surfel / world-space Radiance Cascades). Прошлый GI давал
зернистый, нестабильный свет; surfel'ы получались слишком крупными и их не хватало по
плотности. Новый подход: **воксилизируем сцену в 3D-текстуру**, затем считаем GI
(Radiance Cascades) трассировкой по воксельной сетке.

## Почему воксели

- Сцена задана как набор per-instance SDF-импосторов (`uTransform/uKind/uValues/…`).
  В мире есть готовая функция `scene_sdf(p)` = min по всем инстансам + `emission_of(k)`.
- Воксельная сетка даёт **O(1)** запрос occupancy/альбедо/эмиссии в любой точке и
  дешёвую трассировку лучей через DDA (Amanatides–Woo) — без per-instance sphere-trace
  на каждом шаге луча. Это и есть лекарство от зернистости: GI читает плотную решётку,
  а аппаратная трилинейная фильтрация 3D-текстуры сглаживает результат бесплатно.

## Хранилище (решение: 3D-текстуры)

- `voxelAlbedo` — `rgba8unorm`, storage 3D. `rgb` = альбедо ближайшего инстанса,
  `a` = occupancy (1 = воксель пересекает поверхность, 0 = пусто).
- `voxelEmission` — `rgba16float`, storage 3D. `rgb` = эмиссия (для фазы 2).
- Обе создаются с `STORAGE_BINDING | TEXTURE_BINDING`: пишутся compute-пассом
  (`texture_storage_3d<…, write>`), читаются последующими пассами как `texture_3d<f32>`
  (DDA — точный `textureLoad`; GI фазы 2 — `textureSample` с трилинейностью).

Параметры решётки (uniform): `gridOrigin` (vec3, мир), `cellSize` (f32),
`gridDims` (vec3<i32>). Дефолт: origin `(-32,-32,-2)`, `cellSize 0.5`,
dims `(128,128,32)` → покрывает showcase-сцену (x,y∈[-32,32], z∈[-2,14]); 512k вокселей
(albedo 2 МБ, emission 4 МБ).

## Инфраструктура (расширение)

Существующая обвязка (`VariableMeta`/`GPUVariable`/`utils`) знала только sampled-текстуры,
sampler, uniform и storage-**буферы**. Добавлен `VariableKind.StorageTexture`:
- WGSL: эмитится как обычный `var name: texture_storage_3d<fmt, access>` (без адрес-спейса).
- Группа: `mapKindToGroup[StorageTexture] = 2` (тот же индекс, что у StorageWrite —
  один шейдер не использует оба одновременно; даже если бы использовал, биндинги
  раздаются последовательно внутри группы, без коллизий).
- Layout-entry: `{ storageTexture: { access, format, viewDimension } }`.
- `getGPUResource`: путь `type.startsWith("texture")` уже создаёт view; `viewDimension:"3d"`.

## Фаза 1 — воксилизация + дебаг-рендер  ✅ цель этого этапа

1. `voxelResources.ts` — создаёт обе 3D-текстуры из конфигурации (origin/cellSize/dims).
2. `voxelize.shader.ts` (compute, `@workgroup_size(4,4,4)`): 1 тред = 1 воксель.
   Центр вокселя в мире → инлайн `sceneSDF` helpers + локальные `scene_sdf`/`emission_of`
   (копия из бывшего gather, читает те же scene-буферы). `solid = d <= cellSize*0.5*√3`
   (консервативная полудиагональ). Если solid → `textureStore` альбедо+occupancy и
   эмиссию; иначе очистка в 0.
   - Scene-буферы биндятся напрямую к GPUVariable'ам draw-системы (`sceneInstances`),
     как это делал прошлый gather (без копии данных).
3. `voxelDebug.shader.ts` (fullscreen render): реконструкция мирового луча из
   `invViewProj` (near-точка ndc.z=1 → far-точка ndc.z=0, reverse-Z), slab-пересечение с
   AABB решётки, DDA-марш. Первый воксель с `a>0` → заливка его альбедо с простым Lambert
   по нормали грани (ось шага DDA) + эмиссия. Промах → фон.
4. `createVoxelSystem.ts` — владеет текстурами/пайплайнами/бинд-группами; `voxelize(encoder)`
   (каждый кадр; сцена статична — позже можно гейтить) + `debug(encoder)` → output-текстура.
5. `demo.ts` — снос старого GI, present-селектор `"voxel" | "raw"` (клавиши 1/2), GUI-папка
   «Voxel» (показать/скрыть; ambient).

**Точка приёмки:** видно воксельную версию сцены (ступенчатые грани), формы/цвета верны.

## Фаза 2 — GI Radiance Cascades по вокселям  (после приёмки фазы 1)

Контур: мировая решётка зондов (cascade 0); лучи трассируются voxel-DDA по
`voxelEmission`/`voxelAlbedo` (накопление эмиссии + окклюзия до непрозрачного вокселя);
каскады (удвоение интервала и углового разрешения) + merge сверху-вниз; финальный gather —
реконструкция мировой точки из G-buffer, трилинейная выборка радианса из 3D-текстуры →
`lit = albedo*(ambient + GI)` поверх `frame.renderTexture`.

## Ограничения процесса

- НЕ запускать dev-сервер / vite build: WGSL валидируется только в браузере — ошибки
  присылает пользователь.
- НЕ коммитить.
- Конвенции: `ShaderMeta` + `GPUShader` + `GPUVariable` + тег `wgsl`. После каждого шага
  `npx tsc --noEmit` + `npx oxlint` зелёные.
