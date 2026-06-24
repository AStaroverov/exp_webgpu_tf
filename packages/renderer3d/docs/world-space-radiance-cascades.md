# World-Space Radiance Cascades для renderer3d

> План имплементации **world-space RC** на текущем стеке `renderer3d`.
> Цель — устранить zoom-зависимость текущего экранного RC, перенеся идею
> [mxcop/src-dgi](https://github.com/mxcop/src-dgi) (Surfel Radiance Cascades),
> но **без серфелов и без compute** на первом этапе: вместо адаптивных серфелов —
> регулярная **мировая сетка зондов**, вместо BVH — **sphere-trace по SDF-сцене**.
> Серфелы + compute — отдельный поздний этап (§9).

---

## 1. Зачем и почему именно так

### Проблема текущего RC
Текущий конвейер (`ECS/Systems/Lighting/createRadianceCascadesSystem.ts`) строит
каскады зондов в **экранном пространстве**: зонд = пиксель downscale-текстуры
`rcW×rcH`, а длина интервала каскада (`rayInterval`) меряется в **экранных
пикселях**. Поэтому при зуме одна и та же длина интервала покрывает разное
количество мира → дальность распространения света «дышит» с зумом. Прошлая
попытка конвертировать `baseInterval` пиксели→мир пер-фрагментно (`pxPerWorld =
screenH*projScaleY/(2*probeDist)`) **сломала RC** — это был патч поверх
принципиально экранной структуры.

### Решение
Сделать структуру RC **мировой** с самого начала, как в src-dgi:
- **позиции зондов** — в мировых координатах (привязаны к миру/камере, а не к пикселям);
- **длины интервалов** — в мировых единицах, выводятся из телесного угла;
- **направления** — полная сфера через octahedral-маппинг (а не 2D-углы в плоскости экрана).

Тогда zoom-независимость возникает **по построению**: зум меняет лишь то, сколько
мира попадает в один зонд (пространственное разрешение), но НЕ дальность
распространения света и не геометрию интервалов.

### Почему сетка зондов, а не серфелы
src-dgi платит за адаптивность серфелов целой compute-инфраструктурой, которой у
нас нет:

| src-dgi требует | наш стек |
|---|---|
| compute-пайплайны + атомики | только render-pass'ы (fullscreen-фрагменты, ping-pong) |
| hash-grid + prefix-sum + counting-sort | нет |
| стек серфелов на атомиках (spawn/recycle lifecycle) | нет |
| трассировка по BVH из треугольников | геометрии-как-данных нет; есть SDF-импосторы |

Регулярная **мировая сетка зондов** даёт то же главное свойство (мировой RC), но:
- merge между каскадами превращается из «найти 4 ближайших серфела через hash-grid»
  в **тривиальную билинейную интерполяцию по решётке** — это крупное упрощение;
- не нужен lifecycle зондов (spawn/recycle) — сетка детерминирована из позиции камеры;
- всё ложится на существующий `pass()`-механизм fullscreen-проходов и ping-pong текстур.

Цена сетки vs серфелов: зонды стоят и в пустоте (серфелы — только на поверхностях),
и одиночный наземный слой зондов плохо передаёт радиантность на разной высоте
(см. §8 — известное ограничение, лечится height-слоями / серфелами на §9).

---

## 2. Карта соответствий src-dgi → наша адаптация

| src-dgi (surfel) | наш world-probe-grid |
|---|---|
| Surfel (адаптивный диск на поверхности) | Зонд в узле регулярной мировой сетки |
| `surfel_posr` / `surfel_norw` буферы | позиция выводится из индекса текселя (хранить не нужно) |
| Hash-grid + spawn/count/prefix/accel/recycle | **выкинуто** — сетка детерминирована |
| `surfel_rad` / `surfel_merge` (тайлы интервалов) | atlas-текстуры `probeRad` / `probeMerge` (§3) |
| Gather: trace_bvh по интервалу | Gather: `scene_sdf()` sphere-trace по интервалу (§6) |
| Merge: 4 ближайших серфела через hash-grid, веса 1/dist² | Merge: 4 узла грубой решётки, **билинейные** веса (§5) |
| Composite: 4 ближних серфела на пиксель | Composite: билинейная выборка зондов c0 по world-pos пикселя (§7) |
| octahedral направления, ANGULAR_FACTOR=4, SPATIAL_FACTOR=4 | то же (§4) |
| sun/sky на старшем каскаде при промахе | то же, через общий `SunLight` (§6) |

---

## 3. Модель данных

### 3.1 Система координат
Напоминание из `sdf.shader.ts` / `ResizeSystem.ts`: мир **Z-вверх**, XY — наземная
плоскость (footprint'ы в XY, высота вдоль Z, reverse-Z в проекции). Значит сетка
зондов лежит в **плоскости XY**, направления — полная сфера.

### 3.2 Сетка зондов
- `GRID_DIM` — число зондов по стороне cascade-0 (например `128`).
- `CELL0` — мировой размер ячейки cascade-0 (мировые единицы на зонд).
- Начало сетки привязано к камере и **снапнуто к `CELL0`**, чтобы зонды не «ползли»
  при движении камеры:
  ```
  origin.xy = floor(cameraFocus.xy / CELL0) * CELL0
  probeWorldXY(i, j, c) = origin.xy + (vec2(i, j) + 0.5 - GRID_DIM_c/2) * CELL_c
  probeWorldZ           = PROBE_PLANE_Z   // высота наземного слоя (этап 1–3)
  ```
- Каскад `c`: `CELL_c = CELL0 * 2^c`, `GRID_DIM_c = GRID_DIM / 2^c`
  (пространственный фактор 4 на каскад в 2D — как `SPATIAL_FACTOR=4`).

> Покрытие: мировой охват cascade-0 = `GRID_DIM * CELL0`. Если при отдалении камеры
> видимая область больше охвата — есть две опции: (а) растить `CELL0` ступенями
> (теряем разрешение, держим охват), (б) принять, что зонды покрывают центр кадра.
> На этапе 1 берём (б) + достаточно большой `GRID_DIM*CELL0`; масштабирование — позже.

### 3.3 Atlas-текстуры (хранение интервалов)
Как в src-dgi (тайлы `memory_width×memory_width` на серфел), но тайлы разложены по
2D-сетке зондов. Для каскада `c`:
- `tileW_c = DIR0_W * 2^c` — сторона octahedral-тайла направлений
  (угловой фактор 4 ⇒ сторона ×2 на каскад), `DIR0_W` например `4` (16 направлений на c0).
- размер atlas-текстуры каскада `c`:
  `(GRID_DIM_c * tileW_c) × (GRID_DIM_c * tileW_c)`.
  Спатиально реже × ангулярно плотнее ⇒ размер примерно постоянен по каскадам (классика RC).
- формат `rgba16float`: `rgb` = радиантность интервала, `a` = visibility
  (как `merge_intervals` в src-dgi).

Текстуры (добавить в `createRCTextures`, см. §8.1):
- `probeRad` — сырой gather (выход §6);
- `probeMerge` — результат merge сверху-вниз (§5);
- ping-pong не обязателен: merge каскада `c` читает `probeMerge` каскада `c+1` и
  `probeRad`/`probeMerge` своего `c`, пишет в свой слайс `probeMerge`. Если держим
  все каскады в одной atlas-текстуре «лесенкой» — читаем верхний, пишем текущий (без ping-pong).
  Проще: **отдельная текстура на каскад** (массив `CASCADE_COUNT` текстур), как
  сейчас сделано с per-pass `GPUShader`-инстансами.

### 3.4 Octahedral
Нужны `oct_encode`/`oct_decode` в WGSL (в src-dgi — `octahedral.slang`). Скетч:
```wgsl
fn oct_decode(e_in: vec2<f32>) -> vec3<f32> {
    let e = e_in;                       // [-1,1]^2
    var v = vec3<f32>(e.xy, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0.0) {
        v = vec3<f32>((1.0 - abs(v.yx)) * sign(v.xy), v.z);
    }
    return normalize(v);
}
fn oct_encode(d: vec3<f32>) -> vec2<f32> {
    let n = d / (abs(d.x) + abs(d.y) + abs(d.z));
    var e = n.xy;
    if (n.z < 0.0) { e = (1.0 - abs(n.yx)) * sign(n.xy); }
    return e;                            // [-1,1]^2
}
```
направление ячейки тайла `(u,v)` в `tileW`:
`dir = oct_decode(((vec2(u,v) + 0.5) / tileW) * 2.0 - 1.0)`.

---

## 4. Математика каскадов (мировые единицы)

Бранч-факторы как в src-dgi `cascade.slang`:
```
SPATIAL_FACTOR = 4   // зондов /4 на каскад (в 2D = /2 по стороне)
ANGULAR_FACTOR = 4   // направлений ×4 на каскад (×2 по стороне тайла)
```
Интервалы (start..end луча) — **в мировых единицах**, геометрически растут c фактором 4:
```
interval_scale(0) = [0, 4]
interval_scale(c) = [4^c, 4^(c+1)]            // c >= 1
base_len          = max_solid_angle * DIR0_COUNT / (4*PI)   // как base_interval_length
interval(c)       = base_len * interval_scale(c)            // мировые единицы
```
`max_solid_angle` подбирается так, чтобы интервал c0 примерно соответствовал `CELL0`
(условие отсутствия дыр/перекрытий между каскадами — penumbra condition RC).

Слияние интервалов — дословно src-dgi (near над far c visibility):
```wgsl
fn merge_intervals(near: vec4<f32>, far: vec4<f32>) -> vec4<f32> {
    let radiance   = near.rgb + near.a * far.rgb;  // far проходит только если near прозрачен
    let visibility = near.a * far.a;
    return vec4<f32>(radiance, visibility);
}
```

`CASCADE_COUNT` — до тех пор, пока самый старший интервал не покроет диагональ
охвата сетки: `ceil(log4(GRID_DIM)) + 1` (≈ как сейчас считается из диагонали).

---

## 5. Merge: билинейная интерполяция по решётке (главное упрощение)

Проход на каждый каскад `c` от старшего к младшему (`c = CASCADE_COUNT-2 … 0`),
как `surfel-merge` в `global-illumination.cpp` (`for i = N-2 … 0`).

Фрагмент = одна ячейка `(probe(i,j), dir(u,v))` atlas-текстуры каскада `c`. Шаги:

1. **dst world-pos** зонда `(i,j)` каскада `c` → `P = probeWorldXY(i,j,c)`.
2. В каскаде `c+1` (грубее, шаг `CELL_{c+1} = 2*CELL_c`) найти **4 соседних узла**,
   чья ячейка содержит `P`, и **билинейные веса** `w00..w11` из дробной позиции `P`
   внутри грубой ячейки. (Это и есть замена hash-grid поиска «4 ближайших серфела»
   — для регулярной сетки соседи и веса считаются арифметикой, без поиска.)
3. Целевое направление `(u,v)` каскада `c` соответствует блоку `2×2` направлений
   старшего тайла (угловой фактор 4): `src_interval_id = vec2(u,v) * 2`.
4. Для каждого из 4 пространственных соседей: усреднить 4 угловых под-интервала
   через `merge_intervals(near, far)` (near = текущий `probeMerge[c]` этой ячейки,
   far = `probeMerge[c+1]` соседа), затем взвесить билинейным весом соседа.
5. Записать нормированную сумму в `probeMerge[c]` (на старшем каскаде far берётся
   из `probeRad[c+1]`, ниже — из `probeMerge[c+1]`, как в src-dgi).

Скетч ядра merge (фрагментный, на atlas-текстуру каскада `c`):
```wgsl
// frag → (probe i,j) + (dir u,v) каскада c
let P          = probe_world_xy(ij, c);
let near       = textureLoad(probeMerge_c, frag_xy, 0);   // ячейка этого зонда/направления
let coarse     = grid_bilinear(P, c + 1u);                // 4 индекса + веса w[4]
let srcDir     = uv * 2u;                                 // 2x2 угловой блок
var acc        = vec4<f32>(0.0);
var wsum       = 0.0;
for (var k = 0u; k < 4u; k++) {
    var far = vec4<f32>(0.0);
    for (var a = 0u; a < 4u; a++) {                       // 4 под-интервала
        let off = vec2<u32>(a & 1u, a >> 1u);
        let tex = coarse.cacheXY[k] + srcDir + off;
        far += merge_intervals(near, textureLoad(srcTex, tex, 0)) * 0.25;
    }
    acc  += far * coarse.w[k];
    wsum += coarse.w[k];
}
textureStore_or_return(acc / wsum);
```

---

## 6. Gather: sphere-trace по SDF-сцене

Проход на каждый каскад: фрагмент = ячейка `(probe, dir)`; трассируем интервал по
**всей сцене** и пишем `probeRad`.

```wgsl
// 1. декод ячейки
let ij  = frag_xy / tileW_c;            // индекс зонда
let uv  = frag_xy % tileW_c;            // ячейка направления
let ro0 = vec3<f32>(probe_world_xy(ij, c), PROBE_PLANE_Z);
let dir = oct_decode(((vec2<f32>(uv) + 0.5) / f32(tileW_c)) * 2.0 - 1.0);

// 2. интервал в мировых единицах
let iv  = interval(c);                  // [start, end]
let ro  = ro0 + dir * iv.x;
let tmax = iv.y - iv.x;

// 3. sphere-trace по scene_sdf, отслеживая ближайший инстанс
var t = 0.0; var hit = false; var hitInstance = 0u;
for (var s = 0; s < GATHER_STEPS; s++) {      // GATHER_STEPS ~ 32–48
    let h = scene_sdf(ro + dir * t);          // (dist, instance)
    if (h.dist < 0.001) { hit = true; hitInstance = h.instance; break; }
    t += h.dist;
    if (t > tmax) { break; }
}

// 4. радиантность/visibility интервала
var radiance = vec3<f32>(0.0);
var visibility = select(0.0, 1.0, !hit);      // 1.0 = луч прошёл интервал целиком
if (hit) {
    // эмиссия инстанса (как is_emissive в src-dgi) либо albedo для непрямого
    radiance = emission_of(hitInstance);      // material.x>0 ? color.rgb*intensity : 0
} else if (c == CASCADE_COUNT - 1u) {
    radiance = sky_or_sun(dir);               // только на старшем каскаде (см. ниже)
}
return vec4<f32>(radiance, visibility);
```

### 6.1 `scene_sdf` — min по инстансам
Главный мост к их геометрии: scene SDF = минимум по всем инстансам их **локальных**
SDF (трансформируем world-точку в локальное пространство инстанса — вычесть центр,
обратный yaw — и вызвать существующий `sd_shape3d`).
```wgsl
struct Hit { dist: f32, instance: u32 };
fn scene_sdf(p: vec3<f32>) -> Hit {
    var best = Hit(1e30, 0u);
    for (var k = 0u; k < uInstanceCount; k++) {
        let tr = uTransform[k];
        let hz = uHeights[k] * 0.5;
        let center = vec3<f32>(tr[3].x, tr[3].y, tr[3].z + hz);
        let yaw = atan2(tr[0].y, tr[0].x);
        let rel = p - center;
        let lp  = vec3<f32>(rotZ(rel.xy, cos(-yaw), sin(-yaw)), rel.z);
        let d   = sd_shape3d(lp, k, hz);       // тот же helper из sdf.shader.ts
        if (d < best.dist) { best = Hit(d, k); }
    }
    return best;
}
```
- Нужно **переиспользовать helper'ы** `sd_shape3d` / `sd_2d_for_kind` / `extrude` /
  `footprint_half_xy` из `sdf.shader.ts` (вынести в общий wgsl-фрагмент, импортируемый
  обоими шейдерами, чтобы не дублировать).
- Нужны те же **инстанс-буферы** (`uTransform/uKind/uValues/uRoundness/uHeights/uColor` +
  emission `uMaterial`), что `createDrawShapeSystem` уже заполняет каждый кадр —
  значит их надо **экспортировать как общий ресурс** (см. §8.3).

### 6.2 Стоимость и куллинг — триггер этапа compute
`scene_sdf` — это `O(cells × steps × instances)`. На прототипе с десятком-другим
шейпов это ок. Когда инстансов много, линейный `min` по всем — стена. Это и есть
**естественная точка перехода** к §9 (hash-grid/BVH + compute из src-dgi). На §1–3
явно ограничиваем число инстансов и `GATHER_STEPS`; если упёрлись — §9, а не
микрооптимизация линейного цикла.

### 6.3 Sun/sky
Как в src-dgi (`getSkyColor`/`getAnimatedSunDir`) и в текущем `radianceCascades.shader.ts`:
небо/солнце подмешиваются **только на старшем каскаде** при промахе, направление
солнца берём из общего `SunLight` (`SunLight.angle/enabled`), цвета — из `RCParams`
(`sunColor/skyColor/...`). Переиспользовать существующую логику blending'а.

---

## 7. Composite по G-buffer

Замена/расширение `overlay.shader.ts`. Для каждого пикселя экрана:
1. Восстановить **world-pos** из `depthTexture` (reverse-Z) + обратной проекции
   (как в `composite.slang`/`overlay.shader.ts`: `get_pixel_ray` + depth).
2. Прочитать нормаль из `normalTexture` (`a<0.5` → нет поверхности, вернуть фон).
3. **Билинейно** выбрать 4 зонда cascade-0 вокруг `world.xy` (та же `grid_bilinear`).
4. Проинтегрировать направления тайла каждого зонда с косинусом Ламберта
   `max(0, dot(N, dir))` (как двойной цикл по `memory_width` в `composite.slang`),
   взвесить билинейно, нормировать.
5. `lit = albedo * (ambient + radiance)`; нормаль-aware directional bonus можно
   оставить как сейчас (`dirGain`).

```wgsl
let wp = world_from_depth(uv, depth);
let N  = normalize(unpack_normal(textureLoad(normalTex, px)));
let g  = grid_bilinear(wp.xy, 0u);
var rad = vec3<f32>(0.0); var wsum = 0.0;
for (var k = 0u; k < 4u; k++) {
    var probeRad = vec3<f32>(0.0);
    for (var v = 0u; v < tileW0; v++) {
        for (var u = 0u; u < tileW0; u++) {
            let dir = oct_decode(((vec2<f32>(f32(u),f32(v))+0.5)/f32(tileW0))*2.0-1.0);
            let cos = max(0.0, dot(N, dir));
            probeRad += textureLoad(probeMerge0, g.cacheXY[k] + vec2(u,v), 0).rgb * cos;
        }
    }
    rad  += probeRad * g.w[k];
    wsum += g.w[k];
}
rad = rad / wsum * (4.0 * PI / f32(tileW0 * tileW0));   // нормировка телесного угла
let albedo = textureLoad(sceneTex, px).rgb;
return vec4<f32>(albedo * (uAmbient + rad), 1.0);
```

---

## 8. Точки интеграции (что трогаем)

### 8.1 `WGSL/createFrame.ts`
Добавить в `createRCTextures` массив atlas-текстур каскадов (`probeRad[c]`,
`probeMerge[c]`) с размерами из §3.3. `litTexture` остаётся canvas-sized.
Старые `emission/seed/JFA/df/cascA/cascB` можно оставить (этап 1 идёт параллельно
текущему RC для сравнения), убрать на этапе 3 после валидации.

### 8.2 Новая система `ECS/Systems/Lighting/createWorldRadianceCascadesSystem.ts`
Зеркалит структуру `createRadianceCascadesSystem` (тот же `build()` + `pass()` +
`run()` паттерн, per-pass `GPUShader`-инстансы, `writeScalar`/`writeVec4`-хелперы).
Проходы `run()`:
```
gather:   c = 0 … N-1   (probeRad[c])           // §6, fullscreen на atlas каскада
merge:    c = N-2 … 0   (probeMerge[c])         // §5
composite: probeMerge[0] + G-buffer → litTexture // §7
```
(emit/seed/JFA/df больше не нужны — gather сам трассирует сцену.)

Новые шейдеры рядом: `worldGather.shader.ts`, `worldMerge.shader.ts`,
`worldComposite.shader.ts` + общий `sceneSDF.wgsl` (вынесенные helper'ы из §6.1).

### 8.3 Экспорт инстанс-буферов из `createDrawShapeSystem.ts`
Gather'у нужен доступ к тем же storage-буферам, что заполняет `prepare()`. Вынести
их как общий ресурс: вернуть из `createDrawShapeSystem` ссылки на `GPUVariable`
инстансов (`transform/kind/values/roundness/heights/color/material`) + актуальный
`instanceCount`, и забиндить во второй bind-group gather-шейдера. Helper'ы SDF
вынести в импортируемый wgsl-фрагмент, чтобы оба шейдера делили один код.

### 8.4 `demo.ts`
Завести `createWorldRadianceCascadesSystem({ device, frameTextures, sceneTexture,
depthTexture, normalTexture, sceneInstances })`; в кадре вызвать `worldRc.run(encoder, delta)`
вместо (или рядом с) текущего `rc.run(...)`; `present(encoder, worldRc.outputTexture)`.
Параметры `CELL0/GRID_DIM/DIR0_W/...` — в lil-gui рядом с текущей RC-панелью.

---

## 9. Этапы и критерии валидации

**Этап 1 — мировая сетка зондов, один gather (без каскадов).**
Один каскад (c0), наземный слой зондов, gather sphere-trace, прямой composite без
merge. *Критерий:* сцена освещается зондами; **при зуме дальность/паттерн света
не меняется** (главная цель — снять zoom-зависимость). Сравнить визуально с текущим
экранным RC на одной сцене при разных зумах.

**Этап 2 — каскады + merge сверху-вниз.** §4–§5. *Критерий:* мягкие тени/penumbra
без концентрических колец и без швов между каскадами; стоимость каскадов ≈ постоянна.

**Этап 3 — composite по G-buffer + чистка.** §7, перенос directional/ambient из
`overlay.shader.ts`, удаление неиспользуемых emit/seed/JFA/df-текстур. *Критерий:*
нормаль-aware освещение, паритет качества со старым RC при включённой
zoom-независимости.

**Этап 4 (позже) — высота и/или серфелы.**
- (a) **Height-слои:** превратить наземную сетку в тонкий 3D-объём (несколько
  Z-слоёв), composite — трилинейно. Лечит §8-ограничение (свет на высоте).
- (b) **Серфелы + compute:** когда `scene_sdf` линейный `min` станет стеной (§6.2) —
  ввести compute-пайплайны, hash-grid и lifecycle серфелов из src-dgi. Это
  отдельный крупный трек (нужна compute-инфраструктура, которой в `renderer3d` нет).

---

## 10. Риски и заметки

- **Стабильность во времени:** при снапе `origin` к `CELL0` данные зондов сдвигаются
  на тексель — на §1 пересчитываем каждый кадр без аккумуляции (stateless, как
  текущий RC). Темпоральная репроекция (сдвиг-копия) и jitter/temporal из src-dgi
  (`JITTER`/`TEMPORAL`) — опциональная оптимизация позже.
- **Высота наземного слоя (§8 limitation):** наземные зонды дают радиантность у
  основания объектов; для высоких объектов корректность падает → этап 4a.
- **`GATHER_STEPS` vs дальность:** интервалы старших каскадов длинные — следить, что
  шагов хватает на их `tmax`; иначе пропуски геометрии. Бюджетировать на инстанс-сцене.
- **Нормировка телесного угла** в gather/composite (`4π / dir_count`) — выверить, иначе
  яркость поедет с числом направлений между каскадами.
- **Reverse-Z / Z-up** — все восстановления world-pos и проекции должны совпадать с
  `ResizeSystem.viewProjMatrix` (как уже делает `overlay.shader.ts`).

---

## Источники
- [mxcop/src-dgi](https://github.com/mxcop/src-dgi) — Surfel Radiance Cascades (Vulkan/Slang); ключевые шейдеры: `surfels/{cascade,gather,merge,composite,spawn,recycle,accelerate}.slang`, оркестрация `stages/global-illumination.cpp`.
- Текущий экранный RC: `ECS/Systems/Lighting/createRadianceCascadesSystem.ts` + `*.shader.ts`.
- SDF-импосторы и scene-SDF helper'ы: `ECS/Systems/SDFSystem/sdf.shader.ts`.
