# Obstacles — план реализации (привязка к гексовой сетке)

> Цель: добавить в игру разные препятствия (obstacles), привязанные к гексовой
> сетке. Препятствие занимает **набор гексов** (один или связный кластер).
> Занятость синхронизируется с картой (`HexGrid`). Перекрытие гекса >50% мы НЕ
> измеряем — оно закладывается при генерации: выбираем гексы, занимаем их и грубо
> заполняем их частями больше чем наполовину.

## Принятые решения

| # | Вопрос | Решение |
|---|--------|---------|
| 1 | Семантика занятости гекса | Только `occupy`. `walkable` из `HexCell` **убрать**. У occupant'а хранить `eid` + `worldId` + новый **`kind`** (`Unit`/`Obstacle`). Правило «1 клетка ↔ 1 сущность» необязательно: один `parentEid` занимает несколько клеток. |
| 2 | Подсчёт перекрытия >50% | **Не считаем.** Футпринт (набор гексов) — это *вход*. «>50%» — цель грубой генерации, в неё «верим». Никакого sampling/геометрии. |
| 3 | Форма футпринта | **Связный кластер 1..N** гексов: flood от якоря по соседям. Камень = 1, здание = 3–7. |
| 4 | Заполнение частями | **По типу**: камни — независимые кляксы по каждому гексу (честно >50% на гекс); здания — единый грид по объединению гексов кластера. |
| 5 | Разрушение / освобождение гексов | **Отложить.** Препятствия статичны. `ObstacleFootprint` на родителе храним, систему `releaseHexes` сейчас не пишем. `Hitable`/`Damagable` навешиваем на будущее. |
| 6 | Расстановка при старте | **Двухфазная**: пребилд виртуальной раскладки по сетке → проверка «нормально ли» (связность свободных клеток) → коммит реальных сущностей. |

## Что уже есть (переиспользуем, НЕ пишем заново)

- **Компоненты**: `Obstacle` (тег), `Hitable`, `Damagable`, `Parent`, `Children`
  — `src/Game/ECS/Components/`.
- **Создание тел**: `createRectangleRR` / `createCircleRR`
  — `src/Game/ECS/Components/RigidRender.ts` (рендер-shape + Rapier rigid body +
  `RigidBodyRef`/`RigidBodyState`/`Impulse`/`TorqueImpulse`).
- **Физика**: `createRigidRectangle`, `RigidBodyType.Fixed`,
  `CollisionGroup.OBSTACLE` — `src/Game/Physical/createRigid.ts`, `Config/physics.ts`.
- **Конфиг**: `RockConfig`, `BuildingConfig` — `src/Game/Config/obstacles.ts`.
- **Карта**: `HexGrid` (`hexToWorld`/`worldToHex`/`cornersOf`/`distance`/
  `forEachCell`/`neighbors`/`occupy`/`vacate`/`getOccupant`/`isPassable`),
  `MapWorldId.Game`, синглтон `MapDI.grid` — `src/Game/Map/`.
- **Z-index**: `ZIndexConfig.Rock` (1), `ZIndexConfig.Building` (2).
- **findPath**: A* поверх `isPassable` — `src/Game/Map/findPath.ts`.

> Эталон генерации форм — `packages/tanks`:
> `Entities/Rock/{Rock,generateGridRockShape,RockParts}.ts`,
> `Entities/Building/{Building,generateBuildingShape,BuildingParts}.ts`.
> Логику генерации форм портируем; новое — это привязка к гексам и двухфазная расстановка.

---

## Изменение API карты — `Map/HexGrid.ts` + `Map/HexConfig.ts` (Решение 1)

Рефактор `HexGrid` (затрагивает существующих потребителей — см. ниже):

1. **Убрать** из `HexCell` поле `walkable`; удалить методы `isWalkable`/`setWalkable`.
2. **Добавить** перечисление типа занявшего и поле в occupant:

```ts
export enum OccupantKind {
  Unit = 0,      // подвижная сущность (танк)
  Obstacle = 1,  // статичное препятствие
}

export type HexCell = {
  readonly q: number;
  readonly r: number;
  occupantEid: EntityId | null;
  occupantWorldId: MapWorldId | null;
  occupantKind: OccupantKind | null;
};
```

3. **`occupy`** получает `kind`:

```ts
occupy(q, r, eid, kind: OccupantKind, worldId: MapWorldId = MapWorldId.Game): void
getOccupant(q, r): { eid; worldId; kind } | null
```

4. **`isPassable`** упрощается: `cell != null && cell.occupantEid === null`
   (без проверки walkable).

5. `vacate` без изменений (обнуляет eid/worldId/kind).

**Потребители, которые надо поправить:**
- `createGame.spawnDemoTanks` — `grid.occupy(q, r, tankEid)` → добавить
  `OccupantKind.Unit`.
- `Actions/.../createMoveToHexActionSystem` — там, где `occupy`/`vacate` при
  движении, передать `OccupantKind.Unit`.
- `findPath.ts` — опирается на `isPassable`, менять не нужно (логика «занято →
  блокировано» сохраняется через occupant).
- Любые упоминания `walkable`/`isWalkable`/`setWalkable` в коде/комментариях.

---

## Архитектура расстановки (Решение 6): prebuild → validate → commit

Чистое разделение «планирование (данные) → проверка → создание (ECS)». Планирование
не зависит от ECS-world — это важно и для требования «можно другой world».

```
spawnObstacles({ grid } = MapDI, world = GameDI.world):
  1) PREBUILD  — построить список «заявок» ObstaclePlan на виртуальной занятости
  2) VALIDATE  — проверить, что итоговая раскладка «нормальна»
  3) COMMIT    — для принятых заявок создать сущности и occupy реальной сетки
```

### Типы планирования

```ts
type ObstacleKind = 'rock' | 'building';

type ObstaclePlan = {
  kind: ObstacleKind;
  anchor: { q: number; r: number };
  cells: Array<{ q: number; r: number }>;   // связный кластер (футпринт)
};
```

### 1) PREBUILD — выбор кластеров на виртуальной занятости

Виртуальная занятость = `Set<"q,r">`, инициализируется уже занятыми гексами
(танки), чтобы их не трогать:

```ts
const reserved = new Set<string>();
grid.forEachCell((c) => { if (!grid.isPassable(c.q, c.r)) reserved.add(`${c.q},${c.r}`); });
```

Идём по фиксированной сетке (рокет-сайнс не нужен), для каждой свободной клетки с
вероятностью из `ObstacleConfig` начинаем заявку:
- выбрать тип (`rock` 1 гекс / `building` 3–7 гексов) по `typeWeights`;
- **вырастить связный кластер** `growCluster(anchor, size, reserved)`:
  BFS/flood от якоря, на каждом шаге берём случайного свободного соседа
  (не в `reserved`), пока не наберём `size` или не упрёмся;
- если кластер набрался — пометить его клетки в `reserved`, добавить `ObstaclePlan`.

### 2) VALIDATE — «нормально ли получилось»

Дёшево и достаточно для прототипа: **связность свободных клеток**.
- После пребилда собрать множество свободных гексов (все минус `reserved`).
- Flood-fill от любой свободной клетки; если обойдены ВСЕ свободные — раскладка ОК.
- Если карта раскололась на изолированные карманы — отбраковать последние заявки
  (или перегенерировать с другим seed / меньшей плотностью) и проверить снова.

> Параметры (плотность, число попыток) — в `ObstacleConfig`. Поскольку сетка
> фиксированная 12×12, это копеечные проверки.

### 3) COMMIT — создание сущностей

Для каждого принятого `ObstaclePlan`:
- `createRock(plan, { world })` или `createBuilding(plan, { world })`
  (фабрика получает уже готовый список `cells`!);
- фабрика создаёт родителя + части, навешивает компоненты и **занимает гексы**:
  `for cell of plan.cells: grid.occupy(cell.q, cell.r, parentEid, OccupantKind.Obstacle, worldId)`
  и записывает их в `ObstacleFootprint`.

---

## Новые / изменённые файлы

```
src/Game/Map/HexGrid.ts                          // ИЗМ: убрать walkable, добавить OccupantKind + kind
src/Game/Map/HexConfig.ts                         // (без изменений, либо перенос OccupantKind)
src/Game/createGame.ts                            // ИЗМ: occupy(...Unit); вызвать spawnObstacles()
src/Game/Config/obstacles.ts                      // ИЗМ: + ObstacleConfig (веса/плотность/clusterSize)
src/Game/ECS/Components/ObstacleFootprint.ts       // NEW: занятые гексы на родителе
src/Game/ECS/createGameWorld.ts                    // ИЗМ: регистрация ObstacleFootprint
src/Game/ECS/Entities/Obstacle/
  ├── OBSTACLES_HEX_INTEGRATION_PLAN.md            // этот файл
  ├── index.ts                                     // реэкспорт фабрик и spawnObstacles
  ├── cluster.ts                                   // NEW: growCluster (flood по соседям)
  ├── Rock.ts                                      // NEW: createRock(plan) — кляксы по гексам
  ├── generateGridRockShape.ts                     // NEW: порт из tanks
  ├── RockParts.ts                                 // NEW: порт из tanks
  ├── Building.ts                                  // NEW: createBuilding(plan) — единый грид
  ├── generateBuildingShape.ts                     // NEW: порт из tanks
  └── BuildingParts.ts                             // NEW: порт из tanks
src/Game/ECS/Systems/Obstacle/
  └── spawnObstacles.ts                            // NEW: prebuild → validate → commit
```

### `Config/obstacles.ts` (дополнить)

```ts
export const ObstacleConfig = {
  /** Вероятность начать заявку на свободной клетке при пребилде. */
  spawnChance: 0.12,
  /** Веса выбора типа. */
  typeWeights: { rock: 0.7, building: 0.3 },
  /** Размер кластера здания (в гексах). Камень всегда 1. */
  buildingClusterRange: [3, 7] as [number, number],
  /** Сколько раз пытаться перегенерировать раскладку при провале validate. */
  maxLayoutAttempts: 5,
} as const;
```

### `Components/ObstacleFootprint.ts` (NEW)

Хранит занятые гексы на **родителе** (для будущего `releaseHexes`). Паттерн как
`Children` (NestedArray + count):

```ts
export const createObstacleFootprintComponent = defineComponent((ObstacleFootprint) => {
  const count = TypedArray.u8(delegate.defaultSize);
  const cells = NestedArray.f64(2 * FOOTPRINT_LIMIT, delegate.defaultSize); // пары (q,r)
  return {
    count, cells,
    addComponent(world, eid) { addComponent(world, eid, ObstacleFootprint); count[eid] = 0; },
    add(eid, q, r) { /* запись пары, ++count, защита от FOOTPRINT_LIMIT */ },
    forEach(eid, fn) { /* итерация занятых (q,r) */ },
  };
});
```

Зарегистрировать в `createGameOnlyComponents` (`ECS/createGameWorld.ts`).

### `Entities/Obstacle/cluster.ts` (NEW)

```ts
/** Flood от якоря: связный набор до `size` свободных гексов (минус reserved). */
export function growCluster(
  grid: HexGrid,
  anchor: { q: number; r: number },
  size: number,
  reserved: Set<string>,
): Array<{ q: number; r: number }>;
```

### `Entities/Obstacle/Rock.ts` (NEW) — заполнение по гексам (Решение 4А)

```ts
export function createRock(plan: ObstaclePlan, { world } = GameDI): EntityId | undefined {
  const grid = MapDI.grid;
  const rockEid = addEntity(world);
  Obstacle.addComponent(world, rockEid);
  addTransformComponents(world, rockEid);
  Children.addComponent(world, rockEid);
  ObstacleFootprint.addComponent(world, rockEid);

  for (const cell of plan.cells) {
    const center = grid.hexToWorld(cell)!;
    // генерим кляксу, ВПИСАННУЮ в inradius гекса (radius*√3/2), заполнение >50%
    const parts = generateGridRockShape({ /* нормировка под inradius */ });
    createRockParts(rockEid, center.x, center.y, parts, color, { density });
    grid.occupy(cell.q, cell.r, rockEid, OccupantKind.Obstacle);
    ObstacleFootprint.add(rockEid, cell.q, cell.r);
  }
  return rockEid;
}
```

### `Entities/Obstacle/Building.ts` (NEW) — единый грид по кластеру (Решение 4Б)

```ts
export function createBuilding(plan: ObstaclePlan, { world } = GameDI): EntityId | undefined {
  const grid = MapDI.grid;
  // bbox объединения гексов кластера (по cornersOf всех cells)
  const bbox = unionBBox(plan.cells.map((c) => grid.cornersOf(c)!));
  const buildingEid = addEntity(world);
  Obstacle.addComponent(world, buildingEid);
  addTransformComponents(world, buildingEid);
  Children.addComponent(world, buildingEid);
  ObstacleFootprint.addComponent(world, buildingEid);

  // один грид частей по bbox; оставляем части, чей центр попал в гекс кластера
  const inCluster = new Set(plan.cells.map((c) => `${c.q},${c.r}`));
  const parts = generateBuildingShape({ /* размеры из bbox */ })
    .filter((p) => {
      const h = grid.worldToHex(p.worldX, p.worldY);
      return h && inCluster.has(`${h.q},${h.r}`);
    });
  createBuildingParts(buildingEid, parts, material, { density });

  for (const cell of plan.cells) {
    grid.occupy(cell.q, cell.r, buildingEid, OccupantKind.Obstacle);
    ObstacleFootprint.add(buildingEid, cell.q, cell.r);
  }
  return buildingEid;
}
```

> `generate*Shape.ts`, `*Parts.ts` берём из tanks почти без изменений: только
> импорты `createRectangleRR`/`CollisionGroup`/`ZIndexConfig` из `unknown`, тела
> `RigidBodyType.Fixed`, группа `OBSTACLE`. Части — `Hitable`/`Damagable` (на будущее).

### `Systems/Obstacle/spawnObstacles.ts` (NEW)

Реализует prebuild → validate → commit (см. выше). `world`/`grid` — параметры с
дефолтами `GameDI.world`/`MapDI`, никакого жёсткого импорта-использования `GameDI`.

### Интеграция в `createGame.ts`

```ts
spawnDemoTanks();   // занимают свои гексы (OccupantKind.Unit)
spawnObstacles();   // NEW: prebuild → validate → commit
```

Демо-движение танков (`pickReachableCell` + `findPath`) автоматически обходит
занятые препятствиями гексы — это и есть живая проверка синхронизации с картой.

---

## Порядок работ (чек-лист)

- [ ] **HexGrid рефактор**: убрать `walkable`/`isWalkable`/`setWalkable`; добавить
      `OccupantKind` и поле `occupantKind`; `occupy(...,kind,...)`;
      `getOccupant` → `{eid,worldId,kind}`; `isPassable` без walkable.
- [ ] Поправить потребителей: `spawnDemoTanks`, `createMoveToHexActionSystem`
      (`occupy`/`vacate` с `OccupantKind.Unit`), убрать ссылки на walkable.
- [ ] `Config/obstacles.ts`: + `ObstacleConfig`.
- [ ] `Components/ObstacleFootprint.ts` + регистрация в `createGameWorld.ts`.
- [ ] `Entities/Obstacle/cluster.ts`: `growCluster`.
- [ ] Порт `generateGridRockShape.ts` + `RockParts.ts` + `Rock.ts`
      (кляксы по гексам, вписаны в inradius).
- [ ] Порт `generateBuildingShape.ts` + `BuildingParts.ts` + `Building.ts`
      (единый грид по кластеру + фильтр по принадлежности гексу).
- [ ] `Systems/Obstacle/spawnObstacles.ts`: prebuild → validate(связность) → commit.
- [ ] Вызов `spawnObstacles()` в `createGame.ts` после `spawnDemoTanks()`.

## Проверка результата

1. Препятствия видны на сетке, физически блокируют танки (Fixed + OBSTACLE).
2. Камни: каждый занятый гекс заполнен >50% (кляксы вписаны в гекс).
3. Здания: цельная структура на связном кластере 3–7 гексов.
4. Все клетки кластера помечены `occupy(..., OccupantKind.Obstacle)`;
   `getOccupant` возвращает корректный `kind`.
5. `validate` гарантирует связность свободных клеток — танки не заперты;
   `findPath`/демо-движение объезжает препятствия.
6. Grid-overlay (`Render/Grid`) и occupancy совпадают визуально.

## Заметки про «другой world» (требование 4)

- Фаза **prebuild/validate** работает только с `HexGrid` (чистые данные) — вообще
  не зависит от ECS-world.
- Фаза **commit** принимает `world` параметром; `occupy` пишет `worldId`. Сейчас
  всегда `MapWorldId.Game` + `GameDI.world`. Для отдельного obstacle-world:
  добавить `MapWorldId.Obstacle`, создать второй bitecs world, передать его и
  `worldId` в `createRock`/`createBuilding`. `HexGrid` уже хранит `occupantWorldId`,
  разрешение occupant между мирами поддержано.

## Отложено (Решение 5)

- Система разрушения `createObstacleDestroySystem` + `releaseHexes(parentEid)`
  (по `ObstacleFootprint`: `vacate` гексов, удаление сущностей). Данные для этого
  (`ObstacleFootprint`, `Hitable`/`Damagable`) закладываем сейчас, систему — позже.
