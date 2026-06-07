# Directional Emission для RC-освещения

> Дополнение к [RADIANCE_CASCADES_INTEGRATION.md](./RADIANCE_CASCADES_INTEGRATION.md).
> Добавляет направленный свет (конус) к существующим эмиттерам RC, сохраняя
> omni как поведение по умолчанию.

## 1. Цель и краткое описание механики

Сейчас каждый `LightEmitter` светит во все стороны (omni): в emission-текстуру
пишется HDR-цвет, а raymarch при попадании луча в эмиттер возвращает этот цвет
без учёта направления. Фары танка (`HeadlightConfig`) выглядят как точечные
лампы, а не как направленный пучок.

**Механика.** Эмиттер может быть направленным (directional). У направленного
эмиттера есть **facing** (направление "вперёд" в мире, берётся из вращения
его трансформа) и **глобальная ширина конуса** `emitCone` (показатель
степени). Свет от такого эмиттера в точку приёмника ослабляется множителем
`pow(max(0, dot(facing, dir_emitter_to_receiver)), emitCone)`. Чем дальше
приёмник от оси конуса — тем темнее; за пределами полусферы (dot < 0) — ноль.

**omni по умолчанию.** Признак направленности кодируется **знаком intensity**
эмиттера: `intensity < 0` → directional, `intensity >= 0` → omni. Все
существующие эмиттеры (вспышки, дым, спайс) передают положительный intensity и
продолжают работать как omni без изменений. Ширина конуса — **одна глобальная**
величина, настраиваемая в GUI, а не per-instance (см. §5).

## 2. Архитектура

### Поток данных

```
LightEmitter.intensity (знак = флаг directional)
   │   createDrawShapeSystem.updateBuffers → intensityCollect[i] (знаковое значение)
   ▼
uIntensity[] (SoA, group 1)                  uTransform[] (вращение = facing)
   │                                              │
   ▼   fs_emit (MRT, 2 attachment'а)              ▼
   ├─ @location(0) color  → emissionTexture (rgba16float, additive)
   └─ @location(1) dir    → emitDirTexture  (rg16float,   blend 'none'/replace)
                               omni  → (0,0)
                               dir   → нормализованный facing, Y-flipped
   ▼
radianceCascades.shader.ts (cascade pass)
   raymarch hit:
     emitDir = textureLoad(emitDirTexture, NEAREST)
     if length(emitDir) > 0.5:                    ← классификатор omni/dir
        f = normalize(emitDir)
        sampleLight.rgb *= pow(max(0, dot(-rayDir, f)), uMisc.z)   ← конус-фактор
```

### Ключевые узлы

- **MRT в GPUShader.** `getRenderPipeline` сейчас жёстко строит ровно один
  color-target (`GPUShader.ts:68-95`) и ключ кэша знает только про первый
  формат (`GPUShader.ts:52`). Добавляется опция `targets[]` и blend-режим
  `'none'` (replace). Это **новый** код-путь, не правка (см. §7, finding C1).
- **Текстура направлений `emitDirTexture`** — второй color-attachment emit-пасса,
  формат `rg16float`, размер `rcW × rcH` (как все RC-текстуры). Хранит
  нормализованный facing эмиттера или `(0,0)` для omni/окклюдеров.
- **Конус-фактор в raymarch.** Применяется в точке попадания луча
  (`radianceCascades.shader.ts:93-97`) — единственное место, где известны
  одновременно направление луча `rayDir`, UV попадания и цвет эмиттера.

## 3. Точная математика конуса (под Y-flipped конвенцию)

Все направления живут в одном **Y-flipped** texture-пространстве.

**Эмиттер (запись facing в `fs_emit`).** Вершинный шейдер emit-пасса flip'ает
Y: `to_final_position` возвращает `vec2(res.x, -res.y)` (sdf.shader.ts:286-288).
Вращение эмиттера — `mat2x2(transform[0].xy, transform[1].xy)` (как в
`fs_shadow`, sdf.shader.ts:242); первый столбец `transform[0].xy` — это локальный
+X ("вперёд") в мировых координатах. Чтобы facing жил в том же Y-flipped кадре,
что и лучи:

```wgsl
let t = uTransform[instance_index];
let dir = normalize(vec2<f32>(t[0].x, -t[0].y));   // мировой +X, Y-flipped
```

Прямая форма `(.x, -.y)` от столбца вращения эквивалентна проекции вектора
через ортографическую `uProjection` (та лишь масштабирует/отражает оси), но
дешевле — берём прямую (finding D3).

**Луч (cascade pass).** Внешний цикл задаёт `rayDir = vec2(cos(angle),
-sin(angle))` (radianceCascades.shader.ts:173) — тоже Y-flipped. Внутри
`raymarch` `rayDir = normalize(rayEnd - rayStart)` (строка 80) указывает
**probe → emitter** (приёмник → источник).

**Свет идёт emitter → receiver**, то есть в направлении `-rayDir`. Конус
"светит ли эмиттер в сторону этого приёмника" — это угол между facing `f` и
направлением на приёмник `-rayDir`:

```wgsl
let cone = pow(max(0.0, dot(-rayDir, f)), uMisc.z);   // uMisc.z = emitCone
sampleLight.rgb *= cone;
```

**Проверка самосогласованности.** Оба вектора (`f` и `rayDir`) Y-flipped в одном
кадре, поэтому `dot` корректен. Фара, смотрящая в мировой +X, освещает приёмники
справа от себя. Знаки сходятся (подтверждено по источнику: emit Y-flip
sdf.shader.ts:288, луч Y-flip radianceCascades.shader.ts:173).

## 4. Per-instance флаг: решение

**Решение: знак `uIntensity` (negative = directional). Новые буферы/SoA/
bind-group entries НЕ добавляются.**

Почему именно знак intensity, а не отдельный буфер/поле `radius`:

- Признак направленности — это **один бит**. Заводить новый storage-буфер +
  запись в bind-group + per-frame upload-путь ради одного бита неоправданно.
- `intensityCollect[i]` уже копирует значение verbatim
  (createDrawShapeSystem.ts:108), детектор `intensityChanges` уже гейтит upload
  (строка 78). Отрицательное значение проходит насквозь без нового кода.
- **Безопасность знака подтверждена по источнику.** Единственный GPU-потребитель
  intensity — `fs_emit`. Discard-условие окклюдера — составное:
  `if (intensity == 0.0 && z_height <= SHADOW_Z_THRESHOLD) { discard; }`
  (sdf.shader.ts:199). Отрицательный intensity `!= 0.0`, поэтому не
  коллизирует ни с discard, ни с трактовкой как окклюдер (finding minor:
  «Occluder-discard condition»).
- Цвет берётся через `abs(intensity)`, чтобы отрицательный флаг не затемнял
  эмиттер: `color = uColor[i].rgb * abs(intensity)`.
- Поле `LightEmitter.radius` (Common.ts) остаётся нетронутым (выделено, но
  нигде не читается — подтверждено; репурпозить не нужно).

`LightEmitter.addComponent(world, eid, i = 1, r = 0)` (Common.ts) принимает
intensity первым аргументом — передача отрицательного значения тривиальна.

## 5. Overlap / blend направлений: решение

**Решение: direction-target использует blend `'none'` (replace) → last-writer-
wins. Per-emitter ширина конуса не вводится — ширина глобальная (`emitCone`).**

- Цвет (`@location(0)`) складывается **аддитивно** (как сейчас). Направление
  (`@location(1)`) **НЕЛЬЗЯ** складывать аддитивно: сумма единичных векторов —
  мусор (укорачивает/перекашивает результирующий вектор). Поэтому blend `'none'`.
- При перекрытии двух эмиттеров в одном texel'е побеждает последний записанный.
  При разрешении RC 0.16× канваса (`rcDownscale = 0.4`, createFrame.ts:29) и
  раздельных частях фары перекрытие эмиттеров маловероятно — допустимо, осознанно
  отложено.
- **Окклюдер поверх directional-эмиттера** (краевой случай, finding minor):
  окклюдер пишет `dir = (0,0)`, что под `'none'` может перетереть единичный
  facing и «потушить» конус в этом texel'е. На разрешении RC это субпиксельный
  артефакт; принимается. Документируется как известное ограничение.
- Глобальная ширина — потому что во всех текущих сценах конус нужен одной ширины
  (фары). Per-emitter `coneExponent` был бы мёртвым полем под этим решением.

## 6. Краевые артефакты: решение

**Решение: `textureLoad` (NEAREST, без сэмплера) + порог `length(emitDir) > 0.5`.**

- На границе directional-эмиттера линейная фильтрация смешала бы единичный facing
  с очищенным `(0,0)`, давая укороченный/перекошенный вектор → искажение конуса.
  `textureLoad` берёт точный texel без интерполяции:

  ```wgsl
  let dirTexel = vec2<i32>(rayUv * uResolution);
  let emitDir  = textureLoad(emitDirTexture, dirTexel, 0).xy;
  if (length(emitDir) > 0.5) { /* directional */ }
  ```

- Порог `length > 0.5` **одновременно** классифицирует omni (`(0,0)`, длина 0)
  vs directional (единичный вектор, длина 1). Это делает 4-й канал `.w`-флага
  избыточным → формат `rg16float`, а не `rgba16float` (вдвое дешевле канал;
  `rg16float` доказанно renderable+samplable здесь — seedA/seedB используют его,
  createFrame.ts:41-51, пайплайны seed/jfa createRadianceCascadesSystem.ts:88,105).
- `textureLoad` не требует сэмплера (обходит уже привязанный `linearSampler`),
  поэтому новых bind-group entries нет.
- RC работает на 0.16× канваса — субпиксельный алиасинг и так размывается.

## 7. Точный список правок по файлам

### renderer/src/WGSL/GPUShader.ts
- Новая опция `targets?: { format: GPUTextureFormat, blend?: 'alpha' | 'additive' | 'none' }[]`.
- `fragment.targets` строится из массива (по одному `{format, blend}` на
  attachment). `makeBlend('none') → undefined` (replace).
- **Ключ кэша (строка 52) должен включать ПОЛНЫЙ дескриптор targets — каждый
  format И каждый blend**, а не только первый. Иначе два пайплайна,
  отличающиеся лишь вторым target'ом или single-vs-MRT, коллизируют (finding C1,
  критично: blend `'none'` direction-target'а vs `'additive'` color-target'а —
  именно различающая ось). Старый single-target путь сохраняется (один элемент
  массива). Bind-group-ключ (`${vertex}-${fragment}-${group}`) от targets не
  зависит — не трогаем.

### renderer/src/WGSL/createFrame.ts
- В `createRCTextures` добавить `emitDirTexture`: формат `rg16float`,
  usage `RENDER_ATTACHMENT | TEXTURE_BINDING`, размер `rcW × rcH`. Вернуть его
  из функции (строка 77).

### renderer/src/ECS/Systems/SDFSystem/sdf.shader.ts
- `fs_emit` возвращает структуру:
  ```wgsl
  struct EmitOutput {
    @location(0) color: vec4<f32>,
    @location(1) dir:   vec2<f32>,
  };
  ```
- Сохранить discard по `dist > 0` и составной occluder-discard
  `intensity == 0.0 && z_height <= SHADOW_Z_THRESHOLD` (строка 199).
- `let intensity = uIntensity[i]; color = uColor[i].rgb * abs(intensity)`.
- `if (intensity < 0.0) { dir = normalize(vec2(t[0].x, -t[0].y)); } else { dir = vec2(0.0); }`.
- Любой проходящий фрагмент (включая окклюдер с intensity == 0, z > threshold)
  пишет в `@location(1)`: окклюдер → `dir = (0,0)` → классифицируется omni →
  без конуса (корректно).

### renderer/src/ECS/Systems/SDFSystem/createDrawShapeSystem.ts
- `pipelineEmit` → `targets: [{ format: 'rgba16float', blend: 'additive' }, { format: 'rg16float', blend: 'none' }]`.
- **НЕТ** изменений bind-group (нового uniform нет). **НЕТ** `uEmitCone`/SoA.
  `intensityCollect[i]` уже несёт знаковое значение (строка 108).

### unknown/.../Lighting/createRadianceCascadesSystem.ts
- Добавить `emitDirTexture` в литерал `rcTextures` (строки 67-75) — **ручное
  копирование по имени, иначе начальный build не увидит текстуру** (finding M:
  recreate). Добавить в `destroyTextures` (строки 243-251). `recreate` уже
  переприсваивает `createRCTextures` (строка 256) — покроет авто (но caveat:
  `recreate` сейчас НЕ вызывается ни одним resize-обработчиком; RC
  пересоздаётся только при смене сценария/render-target — это существующее
  состояние, не регрессия).
- Emit-пасс (`run`, строки 204-213): добавить второй colorAttachment =
  `emitDirTexture.createView()`, `clearValue: [0,0,0,0]`, `loadOp: 'clear'`.
- В `build` (перед `getBindGroup` на строке 142): `rcMeta.uniforms.emitDirTexture.setTexture(rcTextures.emitDirTexture)`
  — **до создания bind-group**, иначе bind-group снимет неустановленную
  текстуру → validation error (finding M).
- **uMisc fold.** Свернуть `firstCascadeIndex` + `enableSun` + `emitCone` в один
  `uMisc: vec4<f32>` (12 → 11 uniform-буферов, освобождая место под emitCone без
  превышения лимита 12 — см. §8). Разметка: `.x = firstCascadeIndex`,
  `.y = enableSun`, `.z = emitCone`, `.w` — резерв.
  - Нужен **новый writer на 4 независимых скаляра** (`writeVec4` пакует
    `rgb*mult + w` — неверная форма). В `build` пишется весь `uMisc`. В
    `setParams` (строки 268-281) — **весь** `uMisc` целиком при каждом вызове
    (включая baked `firstCascadeIndex`, иначе live-tuning затрёт его в ноль).
    `firstCascadeIndex` остаётся build-time (не в GUI); live меняются `enableSun`
    (уже в setParams, строка 274) и новый `emitCone`.
- `RCParams.emitCone: number` (тип, строки 12-27) + `DEFAULT_RC_PARAMS.emitCone: 8`
  (строки 30-45).

### unknown/.../Lighting/radianceCascades.shader.ts
- 4-я текстура: `emitDirTexture: texture_2d<f32>` (3/16 → 4/16 — запас есть).
- Свернуть `uFirstCascadeIndex` + `uEnableSun` в `uMisc: vec4<f32>`
  (`.x`/`.y`/`.z = emitCone`). Обновить два read-site'а: `uEnableSun > 0.5` →
  `uMisc.y > 0.5` (строка 179); `uCascadeIndex > uFirstCascadeIndex` →
  `uCascadeIndex > uMisc.x` (строка 189).
- В hit-ветке `raymarch` (строки 93-97, перед `return`), где `rayUv` и `rayDir`
  в области видимости:
  ```wgsl
  let dirTexel = vec2<i32>(rayUv * uResolution);
  let emitDir  = textureLoad(emitDirTexture, dirTexel, 0).xy;
  if (length(emitDir) > 0.5) {
    let f = normalize(emitDir);
    sampleLight.rgb *= pow(max(0.0, dot(-rayDir, f)), uMisc.z);
  }
  ```

### unknown/.../Vehicle/VehicleParts.ts (строка 173)
```ts
LightEmitter.addComponent(world, eid,
  HeadlightConfig.directional ? -HeadlightConfig.intensity : HeadlightConfig.intensity);
```

### unknown/.../Entities/TurretHeadlight.ts (строка 51) — finding M (MAJOR)
То же самое изменение. Это **луч-трапеция** (Shape.Trapezoid, строка 49) —
именно ему конус нужен больше всего. С глобальным `directional: true` на
`HeadlightConfig` оба места должны получить знак-флип, иначе турельная фара
осталась бы omni:
```ts
LightEmitter.addComponent(world, eid,
  HeadlightConfig.directional ? -HeadlightConfig.intensity : HeadlightConfig.intensity);
```
**LightFlash.ts:31** (`options.intensity`, положительный) — осознанно оставляем
omni (вспышка); прочие omni-эмиттеры (spice 1.5, vfx muzzle/hit 3.0) тоже
положительны → не затронуты.

### unknown/src/Game/Config/vehicles.ts (HeadlightConfig, строки 330-333)
Добавить `directional: true`. Поле `coneExponent` НЕ нужно (ширина глобальна).

### unknown/src/ui/createLightingGUI.ts
- Слайдер `emitCone` (диапазон 0..64), `.onChange(apply)`. `params` —
  `structuredClone(DEFAULT_RC_PARAMS)` (строка 15), поэтому `emitCone` появится
  автоматически. `setParams` должен пере-эмитить `emitCone` в `uMisc` каждый кадр.

### renderer/src/ECS/Components/Common.ts
- **БЕЗ изменений** (`radius` остаётся нетронутым).

## 8. Параметры и GUI

| Параметр | Где | Default | Диапазон GUI | Назначение |
|---|---|---|---|---|
| `emitCone` | `RCParams` / `uMisc.z` | `8` | 0..64 | глобальный показатель степени конуса (больше = уже пучок) |
| `HeadlightConfig.directional` | config | `true` | — | переключает фары между directional и omni |

`emitCone` живёт в `uMisc.z` (свёрнутом vec4), пишется в `build` и live в
`setParams`. Паттерн упаковки скаляров в vec4 уже существует
(`writeVec4` для sunColor/skyColor, createRadianceCascadesSystem.ts:293-300) —
новой абстракции не вводим, только новый 4-скалярный writer для `uMisc`.

## 9. Инкрементальные майлстоуны (DONE-критерии на экране)

**M1 — MRT-рефактор без изменения поведения.** Перевести single-target путь в
`getRenderPipeline` на `targets: [{format, blend}]` **И** свернуть ключ кэша на
полный дескриптор targets в том же шаге (иначе вызовы с `targets` но без старых
скаляров коллизируют). Все вызовы переведены.
*DONE:* сцена рендерится попиксельно идентично текущей (emit/seed/jfa/df/cascade/
overlay без визуальной разницы).

**M2 — второй attachment + текстура направлений.** Добавить `emitDirTexture` в
`createRCTextures` + литерал `rcTextures` + `destroyTextures`; второй
colorAttachment в emit-пассе; `pipelineEmit` с двумя targets; `fs_emit` →
`EmitOutput` со `@location(1) dir`. Пока `dir` всегда `(0,0)`.
*DONE:* сцена по-прежнему идентична (dir не читается), нет validation-ошибок,
emit-пасс компилируется с 2-компонентным `@location(1)` в `rg16float`.

**M3 — facing + конус-фактор.** `fs_emit` пишет facing при `intensity < 0`;
4-я текстура в cascade-шейдере; `uMisc` fold; конус-множитель в hit-ветке.
`emitCone` захардкожен временным числом.
*DONE:* эмиттер с отрицательным intensity, повёрнутый в +X, освещает приёмники
справа от себя; omni-эмиттеры светят как раньше.

**M4 — directional headlights + GUI.** `HeadlightConfig.directional: true`;
знак-флип в VehicleParts.ts:173 И TurretHeadlight.ts:51; слайдер `emitCone` в
createLightingGUI; live-tuning через `uMisc` в `setParams`.
*DONE:* фары танка (включая турельную трапецию) дают видимый направленный пучок,
поворачивающийся вместе с корпусом/башней; слайдер `emitCone` сужает/расширяет
пучок в реальном времени.

## 10. Риски

- **R1 (высокий, finding C1).** MRT-API в GPUShader — новый код-путь; ключ кэша
  обязан кодировать полный targets-дескриптор. Самая недоспецифицированная часть.
  Митигация: M1 делает рефактор + ключ атомарно, с попиксельной проверкой.
- **R2 (verify, finding D1).** `vec2<f32>` фрагмент-выход в `rg16float` target —
  по спецификации валидно (число компонент выхода ≥ числа каналов формата),
  seedA/seedB подтверждают renderability. Проверить компиляцию emit-фрагмента с
  2-компонентным `@location(1)` на железе.
- **R3 (verify, finding D2).** Лимит `maxUniformBuffersPerShaderStage`. Подсчёт
  даёт ровно 12 uniform-буферов в cascade-шейдере (resolution, cascadeCount,
  cascadeIndex, baseRayCount, rayInterval, intervalOverlap, srgb,
  firstCascadeIndex, enableSun, sunAngle, sunColor, skyColor). 13-й превысил бы
  дефолтный лимит 12 → fold обязателен. Проверить фактический лимит устройства:
  если > 12, проще добавить 13-й uniform, чем fold.
- **R4 (низкий).** Ordering: `emitDirTexture.setTexture` перед `getBindGroup` в
  `build`; иначе validation-error. Покрыто §7.
- **R5 (низкий).** Окклюдер поверх directional-эмиттера тушит конус в texel'е
  (см. §5). Субпиксельно на 0.16× RC; принято.

## 11. Осознанно отложено

- **Per-emitter ширина конуса** (`coneExponent` на эмиттер). Сейчас ширина
  глобальна; per-instance поле было бы мёртвым. Потребовало бы нового SoA +
  bind-group entry — не оправдано для текущих сцен.
- **Корректное смешение перекрывающихся направлений.** `'none'` →
  last-writer-wins. На 0.16× RC и раздельных частях фар перекрытие
  маловероятно.
- **Окклюдер-vs-эмиттер overlap.** Тот же last-writer-wins; субпиксельный
  артефакт, не исправляется в этой итерации.
- **Resize-wiring для `recreate`.** Сейчас RC пересоздаётся только на смене
  сценария/render-target; живого resize-хука нет (существующее состояние). Не
  вводим в рамках этой фичи.
- **Внутренняя/внешняя мягкость конуса (inner/outer angle).** Один показатель
  степени `pow(dot, k)` даёт мягкий спад; раздельные углы — позже при
  необходимости.
