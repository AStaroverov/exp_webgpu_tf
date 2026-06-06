# Sound System Migration — packages/unknown

> Исследование и обоснование решений: см. [SOUND_SYSTEM_RESEARCH.md](./SOUND_SYSTEM_RESEARCH.md).
> Этот документ — план миграции: что меняем, в каком порядке, и как проверить каждый шаг.

## Цель и принципы

Привести звуковую подсистему к целевой архитектуре из research-доки, не переписывая её.
Порядок приоритетов:

1. **Correctness first.** Сначала чинятся реальные баги (двойной AudioContext,
   коллизия `onEnded`, двойное затухание по расстоянию, отсутствие рамп → клики/зиппер).
   Только потом — новые фичи (шины, ducking, вариативность).
2. **Lean architecture.** Никаких новых файлов и слоёв ради абстракции. Сохраняем
   существующий двухслойный шов:
   - **audio-слой** (`WebAudioTrack` + `SoundManager` + `Config/sound.ts`) — чистый Web
     Audio, ключи — строковые id;
   - **ECS-слой** (`SoundSystem` + `createTankMoveSoundSystem` + `Components/Sound` +
     `Entities/Sound`) — виртуальные голоса, расстояние до камеры, caps, политика
     «кого играть».

   Третий «policy»-слой осознанно **не вводится**: вся политика (caps, resolution,
   jitter-триггеры, duck-триггер) завязана на ECS-понятия (`SoundType`, `Sound.state/loop`,
   distance-to-camera) и не содержит знаний о Web Audio — её нечего выносить, и второго
   потребителя у такого слоя нет.
3. **API stability.** Публичная поверхность (`spawnSoundAtPosition` /
   `spawnSoundAtParent` / `SpawnSoundOptions`, перечисления `SoundType` / `SoundState`,
   `enableSound`, синглтон-алиас `SoundManager` в barrel, вызов `SoundManager.dispose()`
   в `createGame.ts`) остаётся байт-в-байт совместимой.
4. **Headless-safe.** Звук должен работать без рендера (ML/headless-путь). Поэтому спавн
   звука нельзя класть в render-gated код.

## Текущее состояние vs целевое

| Область | Сейчас | Изменение |
|---|---|---|
| AudioContext | `SoundManager` создаёт свой; `WebAudioTrack` экспортирует мёртвый `createWebAudioTrack`/`getSharedAudioContext` со **вторым** `sharedContext` | **Меняем**: удаляем factory + `sharedContext`. Единственный контекст — в `SoundManager` |
| Граф | `BufferSource → gainNode → masterGain(1.0) → destination` | **Меняем/новое**: добавляем per-type buses + master(0.7) + `DynamicsCompressor` (safety limiter) |
| Конструктор `WebAudioTrack` | `gainNode.connect(ctx.destination)` в конструкторе, затем `connect()` делает disconnect+reconnect | **Меняем**: конструктор оставляет `gainNode` неподключённым; `connect()` — единственная проводка |
| `onEnded` | Коллизия: `SoundManager.play` ставит `inUse=false`, `SoundSystem` перезатирает его на `handleStoppedSounds` (single-slot) | **Меняем**: единственный владелец `onEnded` — `SoundSystem`; пул освобождается через `SoundManager.releaseInstance` |
| Громкость (рампы) | `gainNode.gain.value = v` (зиппер/клики) | **Меняем**: все hot-path изменения через `setTargetAtTime`; старт/стоп — `linearRampToValueAtTime` |
| Spatial model | Двойной: `SoundManager` (1500/200 linear) **и** `SoundSystem` (1000/100 cubic) | **Меняем**: удаляем модель из `SoundManager`; остаётся только cubic в `SoundSystem` |
| first-play volume | На первом кадре `Sound.volume[eid]` **не** учитывается, со 2-го — учитывается | **Меняем**: умножать на `Sound.volume[eid]` и на первом кадре |
| Round-robin сэмплов | `Math.random()` (может повторять подряд) | **Меняем**: round-robin по индексу (per-id счётчик) |
| Pitch/gain jitter | нет | **Новое**: ±5% pitch / ±2dB gain только для one-shot'ов (не для лупов) |
| Ducking | нет | **Новое**: duck шины `engine` на первом голосе `Explosion` |
| Explosion-звук | нет | **Новое**: `SoundType.Explosion`, спавн в `destroyTank` |
| Per-type resolution | нет (везде nearest-N) | **Новое**: per-type правило вытеснения |
| Caps | shoot 8, hit 8 | **Меняем**: shoot 8→6, hit 8→4 |
| DebrisCollect | в enum/CONFIG/SOUND_IDS, но не загружается → `play` возвращает null | **Меняем**: удаляем из enum, SOUND_IDS, CONFIG целиком |
| `_audioIndex` на `Sound` | мёртвое поле (только set/-1) | **Меняем**: удаляем |
| `mapParentToLastSoundTime` | модульный Map, никогда не чистится (утечка между рестартами) | **Меняем**: добавляем reset-хук, чистим на destroy |
| `enableSound` resume | fire-and-forget | **Меняем**: `await` промиса resume |
| Спавнеры/энумы/`SoundManager.dispose()` | публичный API | **Не трогаем** |

## Целевая архитектура

### Audio graph + gain staging

```
AudioBufferSource (playbackRate с jitter для one-shot'ов)
        │
        ▼
  track.gainNode      ← per-voice громкость; рампы через setTargetAtTime (tau 0.02);
        │               старт 0→v через linearRamp 5ms; стоп лупа linearRamp→0 за 30ms
        ▼
  busGain[type]       ← per-type шина; ducking-огибающие живут здесь
        │               resting: engine 0.9 / shoot 0.8 / hit 0.8 / explosion 1.0
        ▼
  masterGain (0.7)    ← единственный мастер; запас по headroom
        │
        ▼
  DynamicsCompressor  ← ТОЛЬКО safety limiter (см. ниже про настройки и взаимодействие)
        │
        ▼
  ctx.destination
```

### Раскладка модулей / слои / направление зависимостей

Зависимости идут только вниз: ECS-слой знает про audio-слой, audio-слой про ECS — нет.

```
ECS-слой (знает про SoundType, Sound.state/loop, distance-to-camera)
  ├─ SoundSystem.ts            виртуальные голоса, caps, per-type resolution,
  │                            policy «кого играть», duck-триггер (вызывает soundManager.duck)
  ├─ createTankMoveSoundSystem.ts  флипает Sound.play/stop у лупа движка
  ├─ Components/Sound.ts        компонент Sound, enum SoundType/SoundState
  └─ Entities/Sound.ts          спавнеры (стабильный публичный API)
        │  вызывает (через строковые id + PlayOptions)
        ▼
audio-слой (чистый Web Audio, ключи — строковые id; знаний об ECS нет)
  ├─ SoundManager.ts           AudioContext, buses+master+compressor, пул инстансов,
  │                            round-robin (по индексу), jitter (one-shot only), duck()
  ├─ WebAudioTrack.ts          обёртка над BufferSource+GainNode; рампы, fadeOutAndStop
  └─ Config/sound.ts           константы: buses/master/limiter/duck/jitter + варианты сэмплов
```

### Где живёт какая политика

| Политика | Где | Почему |
|---|---|---|
| Caps (max голосов на тип) | `SoundSystem` CONFIG | завязано на `SoundType` |
| Per-type resolution (кого вытеснять) | `SoundSystem` | нужны distance/`Sound.state` |
| Distance-to-camera + cubic rolloff | `SoundSystem` | ECS-понятие (CameraState, GlobalTransform) |
| Duck-триггер (когда дакать) | `SoundSystem` | событие = первый голос `Explosion` |
| Duck-огибающая (как дакать) | `SoundManager` (`duck()`) | чистый Web Audio на шине |
| Round-robin + jitter | `SoundManager` / `WebAudioTrack` | per-physical-playback, Web Audio |
| Шины/master/limiter | `SoundManager` (значения в `Config/sound.ts`) | граф = audio-слой |

## Фазы миграции

> Каждая фаза самодостаточна и проверяема. Номера фаз следуют порядку research-доки
> (шины+master+limiter → caps/resolution → вариативность → explosion → ducking).

### Фаза 0 — In-place correctness (без изменений архитектуры)

Чиним подтверждённые баги. Никаких шин, ducking, новых типов.

Файлы и изменения:
- **`WebAudioTrack.ts`**:
  - удалить `createWebAudioTrack` / `getSharedAudioContext` / модульный `sharedContext`
    (мёртвый код, второй AudioContext);
  - в конструкторе **не** делать `gainNode.connect(ctx.destination)` — оставить
    `gainNode` неподключённым; единственная проводка — метод `connect()`;
  - добавить `rampVolume(v)` через `setTargetAtTime(v, ctx.currentTime, 0.02)` для
    hot-path; `setVolume` оставить только для немедленной установки на старте.
- **`SoundManager.ts`**:
  - удалить `maxDistance` / `refDistance` / `getDistance` / `calculateDistanceAttenuation`
    / `isInRange` (двойное затухание); `play()` больше не умножает на расстояние
    (`x/y` остаются в `PlayOptions` для совместимости сигнатуры, но не используются);
  - **удалить блок регистрации `onEnded` внутри `play()`** (строки, где
    `instance.track.onEnded(() => instance.inUse = false)`) — это половина коллизии;
  - добавить публичный `releaseInstance(track)`, который ставит `inUse = false` для
    инстанса данного трека (это то, что `SoundSystem` вызовет из своего `onEnded`).
- **`SoundSystem.ts`**:
  - в `onEnded`-хендлере (сейчас только `handleStoppedSounds(eid)`) **также** вызывать
    `soundManager.releaseInstance(track)` — теперь `SoundSystem` единственный владелец
    `onEnded` и сам возвращает инстанс в пул;
  - first-play volume: умножать на `Sound.volume[eid]` и в ветке первого проигрывания
    (сейчас умножение есть только в update-ветке);
  - убедиться, что **все** пути остановки (drop из topN, `Stopped`, orphan-cleanup,
    `disposeSoundSystem`) проходят через `stopInstance` и корректно чистят
    `activeAudios`/`typeSet` без двойного освобождения.
- **`createGame.ts` `enableSound`**: `await` промис resume контекста. Так как
  `enableSound` уже `async`, нужно протащить промис из `ensureContext` наружу (вернуть
  его) или await'ить шаг resume явно перед `loadGameSounds`. Замечание: это **не**
  «бесплатно» — требует вернуть промис из `ensureContext`; но риск низкий.

**НЕ трогаем в этой фазе**: шины, master 0.7, compressor, Explosion, ducking, jitter,
round-robin, caps, resolution, `mapParentToLastSoundTime`.

**DONE-критерий** (консоль/слух):
- В девтулзах ровно **один** `AudioContext` (нет второго от factory).
- Прострелять много раз подряд: пул shoot не «насыщается» — `play` не начинает
  возвращать null после серии выстрелов (раньше из-за коллизии `onEnded` инстансы
  не возвращались в пул). Проверка: лог количества свободных инстансов не падает к 0.
- Звук выстрела/попадания слышен с учётом переданного `volume` уже на первом кадре
  (раньше первый кадр игнорировал `Sound.volume`).

### Фаза 1 — Шины + master + safety limiter

Файлы:
- **`SoundManager.ts`**: в `ensureContext` построить граф
  `busGain[type] → masterGain(0.7) → DynamicsCompressor → destination`; `load()`
  подключает трек к `busGain[config.bus]`, а не напрямую к master.
- **`Config/sound.ts`**: добавить таблицы `buses` (resting-уровни), `master`, `limiter`.
- **`SoundSystem.ts`**: при `play` указывать шину по `SoundType`.

**НЕ трогаем**: ducking (фаза 5), jitter/round-robin (фаза 3), Explosion (фаза 4),
caps/resolution (фаза 2).

**DONE-критерий**:
- Граф в консоли соответствует диаграмме (можно залогировать узлы).
- Общая громкость заметно ниже клиппинга при множестве одновременных звуков;
  на пике лимитер срабатывает (см. `compressor.reduction < 0`), но без слышимого
  «насоса» в нормальной сцене (важно: проверить, что движок не «пыхтит» каждый раз,
  когда стреляют — см. риски про порог/master).

### Фаза 2 — Caps + per-type resolution

Файлы:
- **`SoundSystem.ts`** CONFIG: `maxSoundsPerType` shoot 8→6, hit 8→4 (move/explosion
  по необходимости); добавить per-type resolution в момент, когда голосов больше cap:
  - `shoot` → `PreventNew` (новый не стартует, если cap занят);
  - `hit` / `explosion` → `StopOldest`/`Quietest` (вытеснить старейший/тишайший);
  - `move` → `StopFarthest` (текущий дефолт nearest-N).

  Замечание: per-type resolution — это **новая** политика (сейчас везде единый
  nearest-N), а не отвердевание существующей.

**НЕ трогаем**: граф, ducking, jitter, Explosion.

**DONE-критерий**:
- При >6 одновременных выстрелах новые корректно отбрасываются (а не вытесняют),
  «пулемётность» снижена; на слух нет резких обрывов чужих выстрелов.

### Фаза 3 — Вариативность (round-robin + jitter)

Файлы:
- **`SoundManager.ts`**: round-robin по индексу (per-id инкрементируемый счётчик
  `% buffers.length`) вместо `Math.random()`; pitch-jitter (±5%) и gain-jitter (±2dB)
  применять **только к one-shot'ам** (`!loop`) — лупы движка без pitch-jitter
  (иначе «вой»).
- **`WebAudioTrack.ts`**: добавить `setPlaybackRate(rate)` (до `source.start`).
- **`Config/sound.ts`**: таблицы jitter + массивы вариантов сэмплов.

**НЕ трогаем**: граф, caps, ducking, Explosion.

**DONE-критерий**:
- Серия выстрелов/попаданий звучит вариативно (разный pitch/громкость), без
  «машингана»; движок не «воет» (нет pitch-jitter на лупе).
- Замечание: пока арт поставляет по одному сэмплу на тип, реальный слышимый эффект
  даёт **jitter**, а round-robin по индексу — forward-looking (станет слышен при
  появлении нескольких сэмплов).

### Фаза 4 — Explosion sound

Файлы:
- **`Components/Sound.ts`**: добавить `Explosion = 5` в enum (аддитивно); заодно
  удалить `DebrisCollect = 4` из enum **полностью** (иначе случайный
  `Sound`-entity с типом 4 даст `CONFIG.baseVolume[4] === undefined → NaN`); удалить
  поле `_audioIndex`.
- **`SoundSystem.ts`**: добавить `Explosion` в `SOUND_IDS`/`CONFIG`/`loadGameSounds`;
  удалить `DebrisCollect` из `SOUND_IDS`/`CONFIG`.
- **`TankUtils.ts` `destroyTank`**: рядом со `spawnExplosion` (как **отдельный** вызов,
  НЕ внутри `spawnExplosion`) добавить
  `spawnSoundAtPosition({ type: SoundType.Explosion, x: explosionX, y: explosionY, destroyOnFinish: true })`.
  Критично: `spawnExplosion` (`Explosion.ts`) гейтится на `RenderDI.enabled` и no-op'ит
  без рендера; звук должен быть render-независим, поэтому он — сосед, а не вложенный вызов.
- **`createHitableSystem.ts`**: добавить `destroyOnFinish: true` к спавну `TankHit`
  (`spawnSoundAtParent`, throttled) — чтобы hit-entity не накапливались.

**НЕ трогаем**: ducking (фаза 5), граф/caps/jitter (готовы из предыдущих фаз).

**DONE-критерий**:
- Уничтожение танка даёт звук взрыва **и в headless** (запуск с `RenderDI.enabled = false`:
  визуального взрыва нет, но `spawnSoundAtPosition(Explosion)` вызывается — проверяемо
  логом/счётчиком голосов `explosion`).
- Hit-entity больше не «зависают» как `Stopped` — уничтожаются после доигрывания.
- `play('debris_collect')` нигде не вызывается (тип удалён).

### Фаза 5 — Ducking

Файлы:
- **`SoundManager.ts`**: добавить `duck(busId, amount, holdMs)` — огибающая на
  `busGain[busId]` (см. correctness-notes ниже про точные примитивы рамп).
- **`SoundSystem.ts`**: на **первом** голосе `Explosion` за кадр вызвать
  `soundManager.duck('engine', 0.4, ...)`.
- **`createHitableSystem.ts`**: добавить reset-хук для `mapParentToLastSoundTime`
  (см. ниже) и вызвать его из destroy-пути.

**НЕ трогаем**: всё остальное стабильно.

**DONE-критерий**:
- При взрыве движок кратко приглушается (~до 0.4) и плавно возвращается; нет щелчка
  на входе/выходе ducking.
- После рестарта игры первый hit-звук от переиспользованного eid **не** подавляется
  (Map очищен).

### `mapParentToLastSoundTime` — открытый шов (решить в фазе 5)

Map живёт в `createHitableSystem.ts` (модульный, ключ — `parentEid`), пишется при
спавне hit, **никогда не чистится**. На рестарте eid переиспользуются → устаревший
`lastSpawnTime` может подавить первый hit-звук. В `createHitableSystem` **нет** dispose-хука,
а `SoundDI.destroy` его не видит. Решение (обязательно зафиксировать перед тем, как
называть фазу «done»): экспортировать из `createHitableSystem.ts` функцию `resetHitSoundThrottle()`
и вызвать её из destroy-пути игры (`createGame.ts` destroy), **либо** очищать Map в
начале каждого `createHitableSystem()` (вызывается при пересоздании мира). Без явного
механизма утечка остаётся.

## Web Audio correctness notes

> Эти правила **обязательны** — несколько серьёзных находок ревью касались именно
> неправильных примитивов рамп. Используем правильный примитив под каждую задачу.

### Старт голоса (attack), 5ms — нельзя смешивать с per-frame рампой на одном кадре

На первом кадре голоса `play()` делает attack, и тут же per-frame путь вызывает
`setTargetAtTime` к целевой громкости — **два события на одном `AudioParam` в один
`currentTime` коллизируют** (тот же класс бага, что и `onEnded`-коллизия).

Правило: на **первом** кадре использовать только attack-рампу, per-frame обновления
начинать со **второго** кадра. Attack делать через:
```
gain.setValueAtTime(0, now);
gain.linearRampToValueAtTime(targetVolume, now + 0.005);  // 5ms, линейно реально достигает target
```

### Per-frame обновление громкости (hot path)

`setTargetAtTime(target, now, 0.02)` — экспоненциальное приближение, tau = 0.02s.
Подходит для непрерывного слежения за громкостью движущегося источника. Достигает
~63% за 1·tau, ~95% за 3·tau. Это **не** «доезжает за 20ms» — это асимптота.

### Стоп лупа (fadeOutAndStop) — НЕ через setTargetAtTime

`setTargetAtTime` **никогда** не достигает 0 (асимптота) → на `source.stop()` останется
ненулевой gain → щелчок. Плюс per-frame рампы (tau 0.02) висят на таймлайне и
«композируются». Правильная последовательность:
```
gain.cancelScheduledValues(now);
gain.setValueAtTime(gain.value, now);            // якорь на реальном текущем значении
gain.linearRampToValueAtTime(0, now + 0.030);    // линейно реально достигает 0
source.stop(now + 0.035);
```
**Важно**: `stopInstance` вызывается из `SoundSystem` в нескольких местах (drop из topN,
`Stopped`, orphan-cleanup, `disposeSoundSystem`). **Все** пути остановки **лупов**
должны идти через `fadeOutAndStop`, иначе движок будет щёлкать на самом частом сценарии
(танк остановился → `createTankMoveSoundSystem` ставит `Sound.stop`). One-shot'ы можно
останавливать жёстко (они и так доигрывают до конца).

### Ducking-огибающая — разные примитивы на спад и возврат

- **Спад (duck down)**: до 0.4 за ~30ms — `linearRampToValueAtTime` (быстро,
  детерминированно достигает 0.4; `setTargetAtTime` дал бы только ~63% спада за 30ms).
- **Возврат (recovery)**: `setTargetAtTime(resting, t, 0.13)`. Семантика: tau = 0.13s →
  ~95% за ~0.39s, полное оседание ~0.5–0.6s. **Возврат НЕ «завершён» к 400ms** — это
  только 95%-точка асимптоты. Перекрывающиеся взрывы, приходящие на ~400ms, будут дакать
  от частично восстановленного уровня — это допустимо (компаундится корректно), но **не**
  опираться на «к 400ms бас вернулся».

Перед спадом так же якорить: `cancelScheduledValues(now); setValueAtTime(bus.gain.value, now)`.

### Limiter (DynamicsCompressorNode) — это safety, не «выравниватель»

- У `DynamicsCompressorNode` **нет** параметра makeup-gain в принципе. Формулировка
  «no makeup gain» — это не выбор, а единственно возможное состояние узла; компрессор
  **не** поднимает тихие части (research-доко это подтверждает).
- Настройки: threshold −3dB, knee 0, ratio 20, attack 0.003, release 0.25.
- Узел вносит ~6ms lookahead-латентности — для единственного мастер-лимитера это норм.
- **Риск взаимодействия порога и master** (см. раздел «Риски»): master 0.7 + threshold
  −3dB (≈0.707 linear) + knee 0 + ratio 20 = почти жёсткий лимитер, который может
  включаться почти на любом многоголосном моменте и слышимо «пыхтеть» (движок будет
  приседать каждый раз, когда что-то играет — непреднамеренный sidechain-эффект).
  На фазе 1 проверить на слух; при пыхтении — поднять threshold или мягче knee.

### Lifecycle узлов

- `AudioBufferSourceNode` — одноразовый: после `stop()`/окончания пересоздаётся
  (`createAndStartSource`). Это уже так в `WebAudioTrack`.
- Каждый новый source ставит новый `source.onended`; `_onEndedCallback` — single-slot,
  владелец один (`SoundSystem`). Никто другой `onEnded` не регистрирует.
- pause/resume для one-shot'а вычисляет offset как `(currentTime - startTime) % duration`
  — для не-лупа, доигравшего за пределы duration, это даёт неверный offset (ретриггер
  «уже законченного» звука). **Отложено** (см. ниже): в текущем геймплее one-shot'ы не
  ставятся на паузу осознанно; если начнём — переписать pause/resume для one-shot'ов.

### Single-context rule

Ровно один `AudioContext` на всё приложение, создаётся в `SoundManager.ensureContext`
после user-gesture. Никаких фабрик с собственным контекстом (фаза 0 удаляет их).
Hot-path `resume()` (внутри `play`) остаётся fire-and-forget — он **не** надёжен для
точного тайминга при mid-session re-suspend (фон вкладки, iOS Safari). Для прототипа
приемлемо; на точный тайминг scheduling при подвешенном контексте не опираемся.

## Конфиг — финальная форма

В `Config/sound.ts` рядом с существующим `SoundConfig` (его поля `shootBaseVolume`/
`shootVolumePerWidth` остаются — используются в `Bullet.ts`). Тип-алиас
`export type SoundType = typeof SoundConfig` **переименовать** в `SoundConfigShape`
(коллизия имени с enum `SoundType`; алиас никем не импортируется — безопасно).

Примечание про имена: в коде уже **два** `SoundConfig` — значение в `Config/sound.ts` и
интерфейс `SoundConfig` в `SoundManager.ts` (audio-слой). Не путать; новые таблицы
кладём в `Config/sound.ts`.

Целевая форма (значения — из research-доки):

```ts
// Config/sound.ts
export const SoundBuses = {
    // resting-уровни per-type шин
    engine:    0.9,
    shoot:     0.8,
    hit:       0.8,
    explosion: 1.0,
} as const;

export const SoundMaster = { volume: 0.7 } as const;

export const SoundLimiter = {
    threshold: -3,   // dB
    knee:      0,
    ratio:     20,
    attack:    0.003,
    release:   0.25,
    // makeup-gain отсутствует у узла — намеренно safety-only
} as const;

export const SoundDuck = {
    engine: { amount: 0.4, downMs: 30, recoveryTau: 0.13 },
} as const;

export const SoundJitter = {
    // только one-shot'ы; лупы без pitch-jitter
    oneShot: { pitch: 0.05 /* ±5% */, gainDb: 2 /* ±2dB */ },
} as const;

// массивы вариантов сэмплов (round-robin по индексу).
// Пока по одному сэмплу на тип — round-robin no-op, jitter даёт вариативность.
export const SoundVariants = {
    tank_shoot: ['/assets/sounds/tanks/shot/shot.webm'],
    tank_hit:   ['/assets/sounds/tanks/hit/hit1.webm'],
    explosion:  [/* TODO: asset */],
} as const;
```

Маршрутизация шин и caps по `SoundType` — в `SoundSystem` CONFIG (там же distance,
resolution). Cross-type приоритеты **не вводятся** (см. отложенное).

CONFIG `SoundSystem` (целевые числа):
- `maxSoundsPerType`: TankMove 5, TankShoot **6**, TankHit **4**, Explosion (по сцене),
  без DebrisCollect.
- `baseVolume`: TankMove 0.1, TankShoot 0.2, TankHit 0.3, Explosion (подобрать).
- resolution per-type: shoot=PreventNew, hit/explosion=StopOldest/Quietest, move=StopFarthest.

## Отложено сознательно (anti-speculation)

| Что | Почему отложено |
|---|---|
| Третий «policy»-слой (core/policy/adapter) | Политика завязана на ECS-понятия, не содержит Web Audio, нет второго потребителя — выносить нечего. Это та самая спекулятивная абстракция, которую владелец не любит |
| Global voice budget + virtualization tier | Research-доко стейджит это на «когда сцены вырастут»; сейчас per-type caps достаточно |
| Cross-type priority arbiter | Нужен только вместе с global budget; заменён per-type resolution |
| HDR loudness metric | Research-доко: «только если 1–5 недостаточно, скорее всего достаточно» |
| Честный sidechain-метр для ducking | Событийный duck-триггер (первый Explosion) проще и достаточен |
| Loop play-from-elapsed resync | Лупы движка не требуют ресинка фазы; усложнение без выгоды |
| Переписывание pause/resume для one-shot'ов (offset-баг) | One-shot'ы осознанно не ставятся на паузу в текущем геймплее; фиксим, если начнём |
| Multi-sample арт | Зависит от поставки ассетов; инфраструктура round-robin готова |

## Риски

1. **Limiter «пыхтит».** master 0.7 + threshold −3dB + knee 0 + ratio 20 может включать
   лимитер почти постоянно в многоголосных моментах → движок «приседает» при любом
   звуке (непреднамеренный sidechain). Митигировать на фазе 1 на слух: поднять threshold
   и/или смягчить knee. Это самый вероятный слышимый артефакт.
2. **Рампы — неправильный примитив.** Если разработчик применит `setTargetAtTime` для
   стопа лупа или спада ducking — щелчки/неполный спад. Жёстко следовать correctness-notes
   (linearRamp на старт/стоп/спад, setTargetAtTime только на per-frame и recovery).
3. **Коллизия рамп на первом кадре.** Если attack и per-frame update сработают на одном
   `currentTime` — attack «съедается». Per-frame обновления только со второго кадра голоса.
4. **`mapParentToLastSoundTime` утечка.** Без явного reset-хука первый hit-звук после
   рестарта может подавляться. Зафиксировать механизм очистки (фаза 5).
5. **`onEnded` для нетрекаемых one-shot'ов.** Fire-and-forget one-shot'ы
   (`spawnSoundAtPosition` + `destroyOnFinish`: Bullet, новый Explosion) проходят через
   `SoundManager.play`, но в `activeAudios` per-eid они **трекаются** через ту же ветку
   первого проигрывания (`activeAudios.set(eid, ...)` + `onEnded`). Убедиться, что
   `releaseInstance` вызывается именно из этого `onEnded`, иначе пул не освобождается
   (тот же баг, релоцированный). Проверка — DONE-критерий фазы 0 (пул не насыщается).
6. **`await resume()` плумбинг.** Не «бесплатно»: требует вернуть промис из
   `ensureContext` и заменить `loadGameSounds().catch(...)` на awaited-шаг. Риск низкий,
   но это не однострочник.
7. **Mid-session re-suspend.** Hot-path `resume()` fire-and-forget не гарантирует точный
   тайминг ducking-scheduling при подвешенном контексте. Приемлемо для прототипа.
