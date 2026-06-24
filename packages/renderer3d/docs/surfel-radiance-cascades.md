# Surfel Radiance Cascades для renderer3d (canon src-dgi)

> План портирования **Surfel Radiance Cascades** из [mxcop/src-dgi](https://github.com/mxcop/src-dgi)
> на стек `renderer3d`. Серфелы заменяют регулярную сетку зондов (этапы grid-RC 1–3):
> зонды живут **на поверхностях**, а не в объёме, поэтому «зонд внутри объекта»
> невозможен по построению — это и есть мотив перехода.
>
> grid-RC (`createWorldRadianceCascadesSystem`) остаётся включаемым тоггл-ом, пока
> серфелы не догонят по качеству.

---

## 1. Почему серфелы

Регулярная сетка (grid-RC) даёт zoom-независимый мировой RC, но:
- часть зондов неизбежно **внутри геометрии** → чёрные зонды → портят билинейную интерполяцию соседей (см. baseZ=0 баг);
- объём зондов дорогой и грубый по высоте.

Серфелы (src-dgi) спавнятся **на видимых поверхностях** из G-buffer: зонд = диск на поверхности с нормалью. Внутрь тела попасть не может; плотность адаптивна (где видно — там и зонды). Цена — нужна compute-инфраструктура (атомики, hash-grid, prefix-sum, lifecycle), которой в `renderer3d` нет.

---

## 2. Маппинг src-dgi → наш стек

Пайплайн src-dgi (`src/wyre/platform/vulkan/stages/global-illumination.cpp`, порядок load-bearing):

| src-dgi pass (shader) | назначение | наш аналог |
|---|---|---|
| `surfels/spawn.slang` | спавн серфелов из G-buffer (coverage-троттлинг) | **адаптация**: наш G-buffer (depth+normal+albedo), reverse-Z, world-pos из depth |
| `surfels/count.slang` | подсчёт серфелов на ячейку hash-grid | дословно (compute + atomics) |
| `prefix-sum/*.slang` | prefix-sum по счётчикам ячеек | дословно |
| `surfels/accelerate.slang` | вставка серфелов в hash-grid (atomic_dec + list) | дословно |
| `surfels/gather.slang` | трассировка интервалов серфела | **адаптация**: BVH → `scene_sdf` sphere-trace (есть в worldGather) |
| `surfels/merge.slang` | merge каскадов серфелов (4 ближних через hash-grid) | дословно (hash-grid lookup + octahedral 2×2) |
| `surfels/composite.slang` | сбор радиантности на пиксель из 4 ближних серфелов | **адаптация**: world-pos из нашего depth |
| `surfels/recycle.slang` | переработка серфелов (coverage/маркер) | дословно |
| `surfels/direct_draw.slang`, `heatmap.slang` | дебаг-отрисовка | для валидации стадий |

**Две точки адаптации** (остальное scene-agnostic):
- **gather**: вместо `trace_bvh` — наш `scene_sdf` (min по инстансам, sphere-trace). Уже написан в `worldGather.shader.ts` — переиспользуем helper из `sceneSDF.wgsl.ts`.
- **spawn / composite**: world-pos восстанавливается из нашего reverse-Z depth + `inverse(viewProjMatrix)` (как в `worldComposite`), нормаль — из G-buffer normalTexture.

Octahedral, `merge_intervals`, hash-grid, prefix-sum, recycle-эвристика — переносятся как есть.

### Буферы серфелов (как в src-dgi)
- `surfel_stack` : `u32[]` — `[0]` = указатель стека, далее свободные слоты (атомарный аллокатор).
- `surfel_grid`  : `u32[]` — hash-grid: индексы начала списков на ячейку.
- `surfel_list`  : `u32[]` — записи списков (id серфелов по ячейкам).
- `surfel_posr`  : `vec4[]` — xyz позиция, w = радиус² (`w==0` ⇒ слот мёртв; existence-based).
- `surfel_norw`  : `vec4[]` — xyz нормаль, w = recycle-маркер.
- `surfel_rad`   : `RWTexture2D` (или storage) — radiance-кэш интервалов (octahedral-тайлы).
- `surfel_merge` : `RWTexture2D` — слитый radiance-кэш.

(Атомики: `surfel_stack[0]`, `surfel_grid[*]` — `atomic<u32>`.)

---

## 3. Стадии (валидируем по частям)

### Стадия 0 — compute-инфраструктура  ← ТЕКУЩАЯ
Фундамент, от которого зависит всё. Без привязки к серфелам.
- `GPUShader.getComputePipeline(device, entry, opts)` — аналог `getRenderPipeline`.
- `buildShader`: эмитить `var<storage, read_write>` для `VariableKind.StorageWrite` (сейчас баг — эмитит `var`).
- Конвенция: compute-шейдеры объявляют биндинги с `visibility: GPUShaderStage.COMPUTE`.
- Хелпер диспатча (encode compute pass + `dispatchWorkgroups`).
- **Валидация**: smoke-test compute (пишет thread-индексы в storage-буфер, читаем обратно, лог PASS/FAIL).

### Стадия A — данные серфелов + спавн + дебаг-отрисовка
- Буферы серфелов (§2) + инициализация стека.
- `surfel-spawn` из G-buffer (compute): по 1:N пикселям, coverage-троттлинг (пока без hash-grid — простой счётчик/шанс), запись posr/norw, атомарный pop из стека.
- Дебаг-draw: нарисовать серфелы (как `direct_draw`) поверх сцены.
- **Валидация**: серфелы видны, сидят на поверхностях, не плодятся бесконтрольно.

### Стадия B — hash-grid + lifecycle
- `count → prefix-sum → accelerate` (построение hash-grid каждый кадр).
- coverage-троттлинг спавна через hash-grid (`point_coverage`).
- `recycle` (маркер + перекрытие → возврат в стек).
- **Валидация**: стабильный набор серфелов при движении камеры, без переполнения (heatmap-дебаг).

### Стадия C — gather + composite (один каскад)
- `gather`: на каждый интервал серфела sphere-trace `scene_sdf`; запись в radiance-кэш.
- `composite`: на пиксель — 4 ближних серфела через hash-grid, интеграл по octahedral с косинусом.
- **Валидация**: прямой свет на серфелах, **без «внутри объекта»**; сравнить с grid-RC.

### Стадия D — каскады серфелов + merge
- Иерархия серфел-каскадов (спатиальный/угловой бранчинг как в `cascade.slang`).
- `merge` сверху-вниз (4 ближних серфела старшего каскада, octahedral 2×2).
- **Валидация**: полный GI, мягкие тени, дальний свет.

Каждая стадия — отдельный workflow с валидацией между. grid-RC остаётся тоггл-ом до конца Стадии C–D.

---

## 4. Риски / заметки
- **Compute-лимиты WebGPU**: `maxComputeWorkgroupStorageSize`, `maxStorageBuffersPerShaderStage` (8 по дефолту) — следить при числе буферов серфелов.
- **Атомики в WGSL**: `atomic<u32>` только в `storage, read_write` / `workgroup`. Буферы стека/грида объявлять как `array<atomic<u32>, CAP>`.
- **Стоимость gather по SDF**: `scene_sdf` линейна по инстансам (как в grid-RC) — для прототипа ок; при больших сценах = тот же триггер на ускоряющую структуру.
- **Детерминизм**: spawn/recycle используют atomics + хеши кадра — порядок недетерминирован, это нормально (как в src-dgi).
- **Серфелы стартуют как замена**, но grid-RC не удаляем, пока серфелы не дадут паритет — тоггл презента остаётся.

## Источники
- [mxcop/src-dgi](https://github.com/mxcop/src-dgi) — `assets/shaders/surfels/*.slang`, `prefix-sum/*.slang`, оркестрация `stages/global-illumination.cpp`.
- Наш SDF scene helper: `ECS/Systems/SDFSystem/sceneSDF.wgsl.ts`; world-pos reconstruction: `ECS/Systems/Lighting/worldComposite.shader.ts`.
