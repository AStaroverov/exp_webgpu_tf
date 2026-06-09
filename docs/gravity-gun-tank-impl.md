# Имплементация: грави-танк (GravityTank) и грави-волна

Документ для исполнителя. Описывает, как добавить новый тип техники с грави-пушкой и
саму механику грави-волны, не нарушая принципы ECS из `CLAUDE.md` (компонент = данные,
система = поведение, триггер = query; side-effects откладываются на
`destroyFrame`/`spawnFrame`; нет command buffer; relations = eid; горячие циклы без
аллокаций).

Все ссылки даны в формате `path:line` относительно корня репозитория. Где находки
агентов расходятся или молчат — это помечено явно (см. раздел 6).

---

## 1. Цель и дизайн

Грави-пушка наносит не урон, а **силу**. Выстрел — это волна/конус: всё динамическое,
что находится перед стволом (обломки, враги, **союзники**, любые dynamic-тела), получает
импульс по направлению выстрела. **Урон НЕ наносится вообще** — единственная задача оружия
— **сменить позицию** тела (вмазать в стену, спихнуть в зону, сорвать прицел, оттолкнуть
союзника). Это упрощённый вариант: не «подними конкретный кусок», а **импульс по области
перед стволом** за один триггер.

Цель геймплея: заставить сменить позицию. Всё обучается через PPO, и здесь важное
решение: **выстрел грави-пушкой неотличим от обычного выстрела** — это то же действие
`Fire`, action-space НЕ меняется, переобучения не требует (см. раздел 4). Эффект читаем
из observation (смещение/скорость целей — уже в состоянии).

ECS-обрамление (по принципу из `CLAUDE.md`, раздел «The pattern»):

- **Компонент-данные `GravityFirearm`** на турели/корпусе грави-танка — параметры волны
  (полуугол конуса, дальность, сила импульса, коэффициент урона при ударе, перезарядка).
- **Система `createGravityWaveSystem`** — её query (турели с `GravityFirearm`, у которых
  поднят флаг выстрела и не идёт перезарядка) *есть* триггер. Она находит тела в конусе и
  ставит им импульс через существующий компонент `Impulse`.
- **Существующие системы не трогаются по существу**: импульс применяет уже имеющаяся
  `createApplyImpulseSystem` (`createApplyImpulseSystem.ts`). **Урона нет** — никакого
  контактного-урон пути система волны не использует.

Грави-волна **не добавляет компонент целям**; она лишь эмитит `Impulse` на затронутые
тела. `GravityFirearm` маркирует источник (турель/ствол).

---

## 2. Новый тип техники GravityTank

Точный чек-лист, зеркалящий регистрацию RocketTank. Каждая правка — копия прецедента
RocketTank в том же файле.

1. **Enum типа** — `Config/vehicles.ts:15-22`. Добавить `GravityTank = 6` после
   `MeleeCar = 5` (RocketTank = 3 — прецедент). Это идентификатор типа.

2. **Конфиг-объект** — `Config/vehicles.ts:210-235` (`RocketTankConfig`). Создать
   `GravityTankConfig` с идентичной формой: `{ type, engine, size, padding, density,
   colliderRadius, hullSize, turretSize, hullGrid, turretHeadGrid, caterpillarLines,
   caterpillarSize, trackAnchorXMult, turretSpeed, gun: { gunGrid, reloadTime, caliber,
   bulletOffsetYMult }, colors }`. ВАЖНО (gotcha): конфиг **декларативный**, фабрика его
   не читает — значения дублируются в фабрике вручную, держать в синхроне.

3. **switch в `getTankConfig()`** — `Config/vehicles.ts:364-377` (кейсы 372-373 для
   RocketTank). Добавить `case VehicleType.GravityTank: return GravityTankConfig`.

4. **Калибр снаряда** — `Config/weapons.ts:30-35` (`BulletCaliber`, `Rocket = 3`). Здесь
   решение (см. раздел 3): грави-танк **не стреляет обычной пулей**, поэтому новый
   `BulletCaliber` НЕ нужен. Параметры волны живут в `GravityFirearm`, не в
   `BulletCaliberConfig`. Если по ходу решат давать видимый снаряд-визуал — тогда завести
   `BulletCaliber.GravityBeam = 4` и запись в `BulletCaliberConfig`
   (`Config/weapons.ts:65-115`). По умолчанию — без него. **Открытый вопрос, см. раздел 6.**

5. **Скорость поворота турели** — `Config/weapons.ts:125-140` (`TurretSpeedConfig`).
   Переиспользовать существующую (например `heavy`, как RocketTank) либо добавить запись.
   RocketTank ставит `rotationSpeed = 0` (фиксированный пусковой) в `RocketTank.ts:77` —
   для грави-танка скорее всего нужна поворотная турель (наведение конуса), так что взять
   ненулевую скорость класса (`TurretSpeedConfig.heavy`).

6. **Перезарядка** — `Config/weapons.ts:149-164` (`ReloadConfig`, `rocketLauncher: 5000`
   на 163). Добавить запись, например `gravityGun: <ms>` (предложение: 2000–3000 мс).

7. **Фабрика** — создать `ECS/Entities/Tank/Gravity/GravityTank.ts` по образцу
   `ECS/Entities/Tank/Rocket/RocketTank.ts:1-50`. Сигнатура `createGravityTank(opts: {
   playerId, teamId, x, y, rotation, color })`. Внутри (как RocketTank): `resetOptions`,
   задать `partsCount/size/padding/approximateColliderRadius`,
   `options.vehicleType = VehicleType.GravityTank` (прецедент 47),
   `options.engineType` (48), `createTankBase`, `createTankTracks`, `createTankTurret`,
   `createSlotEntities` на каждую группу частей, `fillAllSlots`,
   `createTankExhaustPipes`, `return tankEid`. Грави-танк **сохраняет** `Firearms` с
   `reloadingDuration` (общий механизм перезарядки/готовности, на него смотрит `FireAction`),
   но вместо `BulletEmitter` (нагрузка-пуля) навешивает `GravityFirearm` (см. раздел 3) —
   именно нагрузка переключает эффект курка на волну, ПОЗИТИВНЫМ query, без `Not` (3.2).
   Поворот турели: `options.turret.rotationSpeed = TurretSpeedConfig.heavy` (не 0, в отличие
   от RocketTank).

8. **Геометрия частей** — создать `ECS/Entities/Tank/Gravity/GravityTankParts.ts` по
   образцу `RocketTankParts.ts:1-67`: экспортировать `SIZE, PADDING, DENSITY,
   HULL_COLS/ROWS, hullSet, <gunSet>, cabinSet`, конфиги гусениц и `PARTS_COUNT`.
   GOTCHA: `PARTS_COUNT` (RocketTankParts.ts:64-66) должен точно равняться сумме длин всех
   наборов частей + гусеницы; он кладётся в `options.partsCount` (RocketTank.ts:43) и
   используется `fillAllSlots()`. Рассинхрон — баг аллокации слотов.

9. **Роутер фабрики** — `ECS/Entities/Tank/createTank.ts:1-35`. Добавить:
   `import { createGravityTank }` (рядом с импортами 4-12); член union `TankVehicleType`
   и член `TankOptions` (строка 18) вида
   `{ type: typeof VehicleType.GravityTank } & Parameters<typeof createGravityTank>[0]`;
   `case VehicleType.GravityTank: return createGravityTank(...)` в switch (рядом с 30-31).
   GOTCHA: пропуск кейса — ошибка исчерпывающей проверки union в строгом TS.

10. **Громкость звука** — `ECS/Entities/Vehicle/VehicleBase.ts:12-19` (`volumeByType`,
    RocketTank → 1.0 на 16). Добавить `[VehicleType.GravityTank]: <0..1>`.

11. **Базовая плотность** — `Config/parts.ts:48-55` (`VehicleBaseDensity`, RocketTank
    `35 * 3` на 52). Добавить `[VehicleType.GravityTank]: <base>`. GOTCHA: плотность
    мультипликативна (`VehicleBaseDensity * PartDensityMultiplier[partType]`).

Любой пропущенный `Record<VehicleType, X>` даёт runtime-ошибку (пустой индекс/кейс).

---

## 3. Оружие: грави-волна

### 3.1. Модель данных — компонент `GravityFirearm`

Создать `ECS/Components/GravityFirearm.ts` по образцу `Explodable.ts:23-41` /
`Firearms.ts:7-44` (фабрика `defineComponent`, см. `renderer/src/ECS/utils.ts:17-34`).

Поля (все — типизированные колонки `bitecs`, индекс по `eid`):

- `coneHalfAngle: f32` — полуугол конуса (рад), например `PI/6`.
- `range: f32` — дальность волны в мировых единицах.
- `impulse: f32` — базовая сила импульса (масштабируется по дистанции — ближе сильнее).
  (Поля урона НЕТ — оружие не наносит урон, см. 3.4.)

**Перезарядка/триггер НЕ дублируются здесь** — грави-турель оставляет себе обычный
`Firearms` (его `reloading`/`shouldShoot`/`isReloading` и используются), а `GravityFirearm`
несёт ТОЛЬКО параметры волны. Это и делает «выстрел неотличимым от обычного» (см. 3.2/4).

Методы (тривиальные, как в `Explodable`/`Impulse`): `addComponent(world, eid, settings)`.

Регистрация компонента — `ECS/createGameWorld.ts:61-108`: импорт фабрики + добавить
`GravityFirearm: createGravityFirearmComponent(world)` в `createGameOnlyComponents()`
(и он попадёт в `getGameComponents(world)`, строки 138-144).

**Решение по модели:** параметры волны живут на компоненте источника, а не в
`BulletCaliberConfig`. Причина: грави-волна — не снаряд (нет letящего тела), её эффект
— мгновенный area-импульс. Это прямой аналог `Explodable` (данные на источнике, система
читает их по query).

### 3.2. Путь выстрела — переиспользуем fire/turret-конвейер

Конвейер огня (находки «Tank Weapon Firing Pipeline»):

- Наведение и команда огня идут через `TurretController` (`TurretController.ts:6-27`):
  `setRotation$()` для поворота, `setShooting$()`/`shouldShoot()` для триггера. Это
  использует `FireAction` (`Actions/systems/FireAction.ts`, 3-фазный исполнитель
  AIMING→WAIT_READY→FIRING).
- Спавнер `createSpawnerBulletsSystem` (`createBulletSystem.ts:6-23`) каждый тик
  проверяет турели и при `shouldShoot() && !isReloading()` стартует перезарядку и
  спавнит пулю.

**Корень проблемы (почему наивно тянет на костыль).** Сейчас `Firearms` (`Firearms.ts`)
смешивает ДВЕ роли: общий **механизм перезарядки** (`reloading`/`isReloading`/
`startReloading`/`updateReloading`) и **полезную нагрузку пули** (`caliber`,
`bulletStartPosition`, `setData`). И `createSpawnerBulletsSystem` (`createBulletSystem.ts:6-23`)
делает обе работы разом: тикает перезарядку И зовёт `spawnBullet`. `FireAction`
(`Actions/systems/FireAction.ts`) при этом завязан на `Firearms.isReloading` — общий механизм.
Поэтому, если просто дать грави-турели `Firearms`, она автоматически начнёт спавнить пули; а
гасить это через `Not(GravityFirearm)` в спавнере = триггер по ОТСУТСТВИЮ компонента —
анти-паттерн из `CLAUDE.md`. Так делать НЕ будем.

**Чистое решение — разделить механизм и нагрузку (как `Explodable`):**

1. **`Firearms` оставить = только перезарядка/готовность.** Убрать из него `caliber` и
   `bulletStartPosition` (и `setData`); оставить `reloading` + методы. На это смотрит
   `FireAction` (`FireAction.ts:WAIT_READY/FIRING` — `Firearms.isReloading`) — **не меняется**.
2. **Новый компонент `BulletEmitter { caliber, bulletStartPosition }`** — позитивный маркер
   «курок порождает пулю». `spawnBullet` (`Bullet.ts:102-120`) читает нагрузку из него, а не
   из `Firearms`.
3. **Турели несут механизм + ОДНУ нагрузку:** обычный танк → `Firearms + BulletEmitter`;
   грави-танк → `Firearms + GravityFirearm`. Нагрузки взаимоисключающие.
4. **Системы-эффекта выбираются ПРИСУТСТВИЕМ своей нагрузки (без `Not`):**
   - `createSpawnerBulletsSystem` → query `[VehicleTurret, TurretController, Firearms, BulletEmitter]`.
   - `createGravityWaveSystem` → query `[VehicleTurret, TurretController, Firearms, GravityFirearm]`.
   Обе на условии `shouldShoot() && !Firearms.isReloading()` → `Firearms.startReloading()` +
   свой эффект. Перезарядку тикает каждая для своего набора; двойного счёта нет (на турели
   ровно одна нагрузка).

Так generic-спавнер перестаёт знать про грави: он лишь **сузил query до своей нагрузки**
(`BulletEmitter`), ровно как `createExplodeSystem` живёт на `[Explodable, Destroy]`. Нажатие
курка идёт штатным `FireAction` → `TurretController.shouldShoot()`; различается только эффект.

Сборка турели — `createTankTurret` (`Tank.ts:67-82`): сейчас `Firearms.addComponent` +
`Firearms.setData(... caliber)` + `setReloadingDuration`. После рефактора `setData` уезжает на
`BulletEmitter` (для обычных танков); грави-танк вместо `BulletEmitter` навешивает `GravityFirearm`.

### 3.3. Система применения волны — `createGravityWaveSystem`

Создать `ECS/Systems/createGravityWaveSystem.ts` по образцу
`createApplyImpulseSystem.ts:8-76` / `createExplodeSystem.ts:22-71`.

Логика тика:
1. `query(world, [VehicleTurret, TurretController, Firearms, GravityFirearm])` — триггер.
2. На каждой турели: перезарядку/триггер берём у **`Firearms`** (не у `GravityFirearm`):
   если `!shouldShoot() || Firearms.isReloading()` — пропуск. Иначе `Firearms.startReloading()`
   и эмит волны.
3. Прочитать мировую позицию и угол ствола из `GlobalTransform`
   (`getMatrixTranslationX/Y`, `getMatrixRotationZ` — как `spawnBullet`,
   `Bullet.ts:97-142`).
4. **Найти тела в конусе — через `intersectionWithShape` (выбрано).** Грубый запрос
   `physicalWorld.intersectionWithShape(...)` с `Ball(range)` (или конусной формой) вокруг
   дула (прецедент закомментирован в `createMapSystem.ts:1-30`), затем точная проверка «в
   конусе» по углу отклонения ≤ `coneHalfAngle`. Маппинг коллайдер→`eid` —
   `getEntityIdByPhysicalId` (как в контактном пути `createGame.ts:117-135`). Колбэк/итерацию
   результата писать без аллокаций; накапливать `eid` в **переиспользуемый** scratch-буфер.
   ОТКРЫТЫЙ ВОПРОС (раздел 6): подтвердить реальную сигнатуру `intersectionWithShape` в
   `@dimforge/rapier2d-simd` (callback-форма vs возврат набора). Если метода нет в нужной
   форме — fallback: `query([RigidBodyRef, RigidBodyState])` + проверка дистанции/угла.
5. Для каждого попавшего тела с `RigidBodyRef.id[eid] !== 0`: посчитать импульс по
   направлению ствола, масштабировать по дистанции (ближе — сильнее), и
   **`Impulse.add(eid, fx, fy)`** (`Impulse.ts:17`). Применение делает штатная
   `createApplyImpulseSystem` (через `rb.applyImpulse(vec, true)`). **Урон не начисляется.**

Не трогать сам источник; цели получают только `Impulse` (см. gotcha из ECS-находок:
«gravity wave should NOT add the GravityWave component to targets»).

**Где в порядке кадра.** `gameTick` (`createGame.ts:184-219`):
`spawnFrame` → `physicalFrame` → ... → `destroyFrame`. Внутри `physicalFrame`
(`createGame.ts:104-136`) порядок: control → `execTransformSystem` → `applyImpulses`
(110) → `physicalWorld.step` (112).

Грави-волна **должна эмитить `Impulse` ДО `applyImpulses()` в том же `physicalFrame`**,
чтобы импульс применился в этом же `physicalWorld.step` и не было лага в один кадр
(`CLAUDE.md`: «System order is part of the design… explode() before destroy()»).
Решение: вставить вызов `gravityWave(delta)` в `physicalFrame` **между**
`updateTurretRotation(delta)` (107) и `applyImpulses()` (110) — после того как турель
довернулась этим тиком (актуальный `GlobalTransform` ствола даёт `execTransformSystem`
на 109; значит ставить вызов сразу после строки 109, перед 110). Инстанцировать систему
рядом с `applyJointMotors` (`createGame.ts:100`).

ОТКРЫТЫЙ ВОПРОС: чтение `GlobalTransform` ствола корректно только после
`execTransformSystem` (109). Порядок «после 109, перед 110» это закрывает; перепроверить
на запуске, что трансформ турели уже свежий в этой точке (раздел 6).

### 3.4. Урон — НЕТ (осознанное решение)

Грави-пушка **не наносит урона вообще**. Её единственная задача — **сменить позицию**
тела (врага или союзника). Волна эмитит только `Impulse`; никакой контактный-урон путь
(`Hitable.hit$`, `Damagable`, `drainContactForceEvents`) система волны **не использует**.

Следствие — `GravityFirearm` НЕ несёт `Damagable`/`Hitable`-логики, и в `createTankTurret`
для грави-танка эти урон-компоненты на «оружие» не вешаются. Если отброшенное волной тело
во что-то врежется, штатная физика отработает обычное столкновение как для любого летящего
тела — но это побочный эффект физики, не часть оружия, и специально не усиливается.

Геймплейно урон приходит из *других* источников (пушки союзников, зоны, обломки) — грави
лишь ставит врага в невыгодную позицию. Это и есть «без скрытой магии»: одно оружие = один
эффект (сила), без скрытого урона.

### 3.5. Обломки и группы коллизий

Персистентные обломки появляются через `tearOffTankPart()` (`TankUtils.ts:67-95`):
снимаются `VehiclePart`/`Joint`, рвётся Rapier-джойнт, группа коллизий меняется на
`CollisionGroup.ALL & ~(BASE | BULLET | TURRET parts)` (`physics.ts:1-115`). После этого
обломок — свободное Dynamic-тело и взаимодействует со всеми не-vehicle телами.

Следствия для волны:
- Обломки **уже** Dynamic с `RigidBodyRef` → `intersectionWithShape` из 3.3 их находит,
  `Impulse.add` на них работает (прецедент `applyExplosionImpulse.ts:1-55` толкает именно
  такие отделённые части).
- Урон обломки не несут и не должны (см. 3.4) — волна просто их раскидывает; это и есть
  «расчистить/завалить позицию». Никакого `Hitable` для этого не требуется.
- Волна должна толкать **все** dynamic-тела перед стволом (враги + обломки + любые
  свободные тела), поэтому query из 3.3 фильтрует по `RigidBodyRef`, а НЕ по типу. Это
  ровно anti-pattern-избегание из `CLAUDE.md` (никаких `if (тип)` в общей системе).
  При необходимости исключить своих/пули — `Not(Bullet)` и проверка `TeamRef`.

---

## 4. Слой действий и RL

### 4.1. НИКАКОГО нового действия — переиспользуем `Fire`

**Решение (по уточнению):** выстрел грави-пушкой **неотличим от обычного выстрела**.
Грави-танк использует ровно то же действие `Fire`, что и любой другой танк. Поэтому:

- **НЕ создаётся** новый `ActionKind.GravityFire`.
- **НЕ меняется** `ACTION_DIM_TOTAL` (остаётся 13: `[Hold | MoveStep×6 | Fire×6]`,
  `consts.ts:18-46`).
- **НЕ трогаются** `applyActionToGame.ts`, `computeActionMask.ts`, выходная голова сети
  `models/Networks/v3.ts`.
- **Переобучение из-за action-space НЕ требуется** — форма входов/выходов сети та же.

Агент просто выдаёт `Fire` по направлению; *что* произойдёт (пуля или грави-волна)
определяется **оружием танка** на игровой стороне (`Firearms` vs `Firearms+GravityFirearm`),
а не действием. Это и есть смысл «выстрел не отличается»: различие живёт в ECS-эффекте
курка (3.2), а слой действий/RL о грави-пушке вообще не знает.

> Это работает, потому что (3.2) `FireAction` завязан на общий механизм `Firearms.isReloading()`,
> а *эффект* курка выбирается нагрузкой (`BulletEmitter` vs `GravityFirearm`). Грави-турель
> несёт `Firearms` (механизм) + `GravityFirearm` (нагрузка) — `FireAction` отрабатывает как есть.

### 4.2. Читаемость эффекта в observation

Наблюдение — egocentric 11×11×15 (`state/board.ts:31-109`, `InputArrays.ts:11-26`).
Эффект волны (смещение врага/обломков) **уже читается** существующими каналами:
`CoordX/CoordY` + сам факт перемещения юнита между гексами на борде. Поэтому **новый канал
НЕ добавляем** — это сохранило бы `CHANNELS`/`BOARD_SIZE` и не требует переобучения
(согласовано с решением из 4.1 — форма наблюдения не меняется). Если позже окажется, что
эффект читается слишком косвенно — отдельный канал «недавно получил импульс/скорость»
(из `RigidBodyState.linvel`) можно добавить, но это уже переобучение (раздел 6).

### 4.3. Reward (предложение)

Reward — per-macro-action, дельта `ScoreTracker` (`reward/calculateReward.ts:47-51`),
dense-shaping затухает к концу обучения. Грави-пушка **урона не даёт**, поэтому shaping
строится вокруг **позиционной выгоды**:
- **+ за смещение врага** в нежелательную для него сторону: к стене, в зону поражения,
  прочь от его цели (дельта дистанции врага до укрытия/до своей цели).
- **+ большой за спихивание врага в зону** (out-of-zone destroy — `destroyOutOfZone`).
  Урон/килл, который придёт потом от союзников/зоны, начислится штатно — грави лишь
  создал позицию.
- **(опц.) + за смещение союзника в безопасность/из зоны** — раз оружие действует и на своих.
- **− штраф времени** за no-op выстрел (волна никого не задела) — как штраф времени на
  прочие действия.

---

## 5. Порядок имплементации (мелкие коммиты)

- [ ] **Коммит 1 — регистрация типа (без поведения).** `Config/vehicles.ts` (enum 15-22,
  `GravityTankConfig` 210-235-аналог, switch 364-377), `Config/weapons.ts`
  (`ReloadConfig.gravityGun` 149-164; turret speed — переиспользовать),
  `Config/parts.ts` (`VehicleBaseDensity` 48-55), `VehicleBase.ts` (`volumeByType` 12-19).
- [ ] **Коммит 2 — геометрия и фабрика.** `ECS/Entities/Tank/Gravity/GravityTankParts.ts`
  (образец `RocketTankParts.ts`), `ECS/Entities/Tank/Gravity/GravityTank.ts` (образец
  `RocketTank.ts`), роутер `createTank.ts:1-35` (import + union + case). На этом этапе
  танк ездит и наводится, но не «стреляет».
- [ ] **Коммит 3 — рефактор: вынести нагрузку пули в `BulletEmitter` (без новой фичи).**
  Чисто разделить роли `Firearms` (см. 3.2), не меняя поведение обычных танков:
  `ECS/Components/Firearms.ts` ужать до `reloading` + методов (убрать `caliber`,
  `bulletStartPosition`, `setData`); новый `ECS/Components/BulletEmitter.ts`
  `{ caliber, bulletStartPosition }` + `setData`, регистрация в `createGameWorld.ts:61-108`;
  `spawnBullet` (`Bullet.ts:102-120`) читает из `BulletEmitter`; `createTankTurret`
  (`Tank.ts:80-82`) для обычных танков навешивает `Firearms` + `BulletEmitter`;
  `createSpawnerBulletsSystem` (`createBulletSystem.ts:6-23`) → query
  `[VehicleTurret, TurretController, Firearms, BulletEmitter]`. Проверить, что обычные танки
  стреляют как раньше (регрессия). Это самостоятельный, осмысленный сам по себе коммит.
- [ ] **Коммит 4 — компонент `GravityFirearm`.** `ECS/Components/GravityFirearm.ts`
  (образец `Explodable.ts` — чистые данные волны: cone/range/impulse), регистрация в
  `createGameWorld.ts:61-108`. В `createTankTurret` для GravityTank: `Firearms` (механизм) +
  `GravityFirearm` (нагрузка), БЕЗ `BulletEmitter`.
- [ ] **Коммит 5 — система волны.** `ECS/Systems/createGravityWaveSystem.ts` (образец
  `createApplyImpulseSystem.ts`): query `[VehicleTurret, TurretController, Firearms,
  GravityFirearm]`, на `shouldShoot() && !Firearms.isReloading()` →
  `intersectionWithShape` (конус перед стволом) → `Impulse.add` затронутым телам,
  `Firearms.startReloading()`. Инстанс в `createGame.ts:~100`, вызов в `physicalFrame`
  после `execTransformSystem()` (109), перед `applyImpulses()` (110). **Урона нет.**
- [ ] **Коммит 6 — спавн в сценариях.** Добавить грави-танк в dev-мир (`setupDemoWorld`)
  и/или в обучающий сценарий `ppo_unknown` (по памяти проекта — build-specific контент
  живёт там, не в `createGame`).
- [ ] **(опц.) Коммит 7 — reward-shaping.** Позиционный shaping в
  `reward/calculateReward.ts:47-51` (смещение врага к стене/в зону). НЕ обязателен для
  запуска — действие `Fire` и так работает на грави-танке без правок RL-слоя.

> **Чего в плане НЕТ (по решению из раздела 4):** нового `ActionKind`, правок
> `consts.ts`/`applyActionToGame.ts`/`computeActionMask.ts`/сети, нового канала observation,
> урон-кода. Action-space и форма сети не меняются → переобучение из-за интерфейса не нужно.

---

## 6. Открытые вопросы / риски

1. **Сигнатура `intersectionWithShape` (выбранный путь).** В находках метод показан только
   закомментированным (`createMapSystem.ts:1-30`). Перед использованием подтвердить реальную
   сигнатуру в `@dimforge/rapier2d-simd`: callback-форма vs возврат набора хэндлов, как
   фильтровать по группам/исключать источник. Fallback, если форма неудобна — итерация
   `query([RigidBodyRef, RigidBodyState])` + проверка дистанции/угла.

2. **Свежесть `GlobalTransform` ствола в `physicalFrame`.** Вызов волны ставим после
   `execTransformSystem()` (109). Проверить на запуске, что трансформ турели уже обновлён
   в этой точке (иначе конус будет смотреть по прошлому кадру).

3. **Регрессия рефактора `BulletEmitter` (коммит 3).** Вынос `caliber`/`bulletStartPosition`
   из `Firearms` в `BulletEmitter` трогает работающую пушку. Проверить, что все читатели
   нагрузки переведены (`spawnBullet` `Bullet.ts:102-120`, проставление в `Tank.ts:80-82`) и
   обычные танки стреляют как раньше. Других читателей `Firearms.caliber`/`bulletStartPosition`
   быть не должно — перепроверить grep'ом перед коммитом.

4. **Новый калибр vs нет (раздел 2 п.4).** По умолчанию грави-танк без отдельного
   `BulletCaliber` (волна — не снаряд). Если нужен видимый летящий визуал волны — завести
   `BulletCaliber.GravityBeam` + запись в `BulletCaliberConfig` (`weapons.ts:65-115`).
   Решить на этапе визуала.

5. **Баланс силы/перезарядки/конуса.** `impulse`, `coneHalfAngle`, `range`, `ReloadConfig
   .gravityGun` — все требуют игровой настройки и профилирования. Поскольку урона нет,
   «слишком сильно» = телепортирует врага через пол-карты; «слишком слабо» = бесполезно.
   Профилировать по worst-frame (GC от спавна VFX, если будет визуал волны).

6. **Дружественное смещение.** Оружие толкает и союзников. Это фича (раздел 4.3), но риск
   случайно спихнуть своего в зону. Решить по балансу — нужна ли защита/индикация.

> **Снято по решениям этой итерации:** контактный-урон путь и порог contact-force (урона
> нет); `Hitable` на обломках (не нужен); расширение action-space и переобучение из-за
> интерфейса (выстрел = обычный `Fire`); новый канал observation; маска грави-направлений.
