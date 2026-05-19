# План: разделение `ml` + `ml-common` → `ppo` + `ppo_tanks`

Цель: вынести generic PPO-движок (`ppo`) в отдельный пакет, оставив в `ppo_tanks` только домен (танки, сценарии, награды, конфиг, UI). Шаги атомарны и **каждый завершается зелёным `tsc --noEmit`**.

Легенда: ✏️ = редактирование, 📦 = move/copy файла, 🆕 = новый файл, 🗑️ = удаление.

---

## Фаза 0 — Подготовка инфраструктуры

### 0.1 🆕 Создать каркас пакета `ppo`
- `packages/ppo/package.json` (name `ppo`, type module, без скриптов)
- `packages/ppo/tsconfig.json` (extends корневого)
- `packages/ppo/src/index.ts` — пустой
- Проверка: `tsc --noEmit`.

### 0.2 🆕 Создать каркас пакета `ppo_tanks`
- `packages/ppo_tanks/package.json`
- `packages/ppo_tanks/tsconfig.json`
- `packages/ppo_tanks/src/index.ts` — пустой
- Проверка: `tsc --noEmit`.

### 0.3 ✏️ Зарегистрировать оба пакета в корневом workspace
- Корневой `package.json` → `workspaces` (или `paths` в корневом `tsconfig.json`, как уже сделано для остальных пакетов).
- Проверка: `tsc --noEmit`, `ml` / `ml-common` продолжают собираться.

---

## Фаза 1 — Разрыв цикла `ml-common ↔ ml/src/Models/Create.ts`

### 1.1 🆕 Файл `packages/ppo_tanks/src/models/dims.ts`
Скопировать из `ml/src/Models/Create.ts` **только константы**:
```
TANK_FEATURES_DIM, TANK_HISTORY_STEPS, TANK_HISTORY_FEATURE_DIM,
TURRET_SLOTS, TURRET_FEATURES_DIM,
ENEMY_SLOTS, ENEMY_FEATURES_DIM,
ALLY_SLOTS, ALLY_FEATURES_DIM,
BULLET_SLOTS, BULLET_FEATURES_DIM,
RAY_SLOTS, RAY_FEATURES_DIM,
GRID_SIZE, GRID_CELL_FEATURES, GRID_CELLS,
ACTION_HEAD_DIMS, RAY_HIT_TYPE_COUNT
```
Re-export из `Create.ts` через `export * from 'ppo_tanks/src/models/dims.ts'`, чтобы не сломать существующие импорты.

### 1.2 ✏️ Переключить импорты dims в `ml-common`
Файлы:
- `ml-common/InputArrays.ts`
- `ml-common/InputTensors.ts`
- `ml-common/applyActionToTank.ts`

Меняем `from '../ml/src/Models/Create.ts'` → `from '../ppo_tanks/src/models/dims.ts'`.
Цикл разорван.

### 1.3 ✏️ Удалить дублирующие константы из `ml/src/Models/Create.ts`
Оставить только функции `createPolicyNetwork`/`createValueNetwork`/`shouldNoiseLayer`. Константы только re-export.
Проверка: `tsc --noEmit`.

---

## Фаза 2 — Перенос generic RL-утилит в `ppo`

Каждый под-шаг: 📦 move + ✏️ обновить импортирующие файлы. Делать **по одному файлу**.

### 2.1 📦 `ml-common/Tensor.ts` → `ppo/src/utils/Tensor.ts`
### 2.2 📦 `ml-common/flat.ts` → `ppo/src/utils/flat.ts`
### 2.3 📦 `ml-common/logProb.ts` → `ppo/src/utils/logProb.ts`
### 2.4 📦 `ml-common/getDynamicLearningRate.ts` → `ppo/src/utils/getDynamicLearningRate.ts`
### 2.5 📦 `ml-common/modelsCopy.ts` → `ppo/src/utils/modelsCopy.ts`
### 2.6 📦 `ml-common/utils.ts` (network settings) → `ppo/src/utils/networkSettings.ts`
### 2.7 📦 `ml-common/analyzeVTrace.ts` → `ppo/src/metrics/analyzeVTrace.ts`
### 2.8 📦 `ml-common/Memory.ts` → `ppo/src/memory/Memory.ts`
### 2.9 📦 `ml-common/ReplayBuffer.ts` → `ppo/src/memory/ReplayBuffer.ts`
### 2.10 📦 `ml-common/PrioritizedReplayBuffer.ts` → `ppo/src/memory/PrioritizedReplayBuffer.ts`
### 2.11 📦 `ml-common/ColoredNoise.ts` → `ppo/src/noise/ColoredNoise.ts`
### 2.12 📦 `ml-common/ColoredNoiseApprox.ts` → `ppo/src/noise/ColoredNoiseApprox.ts`
### 2.13 📦 `ml-common/DirichletNoise.ts` → `ppo/src/noise/DirichletNoise.ts`
### 2.14 📦 `ml-common/NoiseMatrix.ts` → `ppo/src/noise/NoiseMatrix.ts`
### 2.15 📦 `ml-common/initTensorFlow.ts` → `ppo/src/infra/initTensorFlow.ts`
### 2.16 📦 `ml-common/console.ts` → `ppo/src/infra/console.ts`
### 2.17 📦 `ml-common/unhandledErrors.ts` → `ppo/src/infra/unhandledErrors.ts`
### 2.18 📦 `ml-common/channels.ts` → `ppo/src/infra/channels.ts` (forceExitChannel + metricsChannels)

**После 2.18**: `ml-common` содержит только домен (Curriculum, InputArrays, InputTensors, applyActionToTank, compute*, debug, uiUtils, Metrics, config, consts).

---

## Фаза 3 — Перенос generic слоёв и оптимизаторов в `ppo`

### 3.1 📦 `ml/src/Models/Layers/**` → `ppo/src/models/Layers/**`
### 3.2 📦 `ml/src/Models/Optimizer/**` → `ppo/src/models/Optimizer/**`
### 3.3 📦 `ml/src/Models/ApplyLayers.ts` → `ppo/src/models/ApplyLayers.ts`
### 3.4 📦 `ml/src/Models/def.ts` → `ppo/src/models/def.ts`
### 3.5 📦 `ml/src/Models/Utils.ts` → `ppo/src/models/Utils.ts`
### 3.6 📦 `ml/src/Models/Transfer.ts` → `ppo/src/models/Transfer.ts`
### 3.7 📦 `ml/src/Models/Restore.ts` → `ppo/src/models/Restore.ts`
### 3.8 ✏️ Удалить `CONFIG.savePath` обращение из `Transfer.ts`/`Restore.ts`
Параметризовать функцию: `saveNetworkToDB(network, modelName, savePath)`. Все вызовы получают `savePath` извне.

---

## Фаза 4 — Декаплинг PPO core

### 4.1 ✏️ Параметризовать `Memory`/`AgentMemoryBatch` по типу состояния
В `ppo/src/memory/Memory.ts`:
- `AgentMemory<S>`, `AgentMemoryBatch<S>`, `PreparedBatch<S>` — generic.
- Дефолт `S = unknown`.
Обновить все импортирующие места — пока `S = InputArrays` (импортим через type alias из `ppo_tanks`).

### 4.2 🆕 `ppo/src/core/StateBindings.ts`
```ts
export interface StateBindings<S> {
  createInputTensors(batch: S[]): tf.Tensor[];
  prepareRandomInputArrays(): S;
}
```

### 4.3 ✏️ Вытащить `ACTION_HEAD_DIMS` и `ACTION_DIM` из core
- `ml/src/PPO/train.ts` → принимает `actionHeadDims: number[]` в `computeRetraceTargets` и в `trainPolicyNetwork` (для `tf.tensor2d(... [B, actionDim])`).
- `createPolicyLearnerAgent.ts` → target-entropy считается снаружи и передаётся как `targetEntropy: number`.
- Удалить импорт `ACTION_DIM` из `ml-common/consts.ts` внутри core.

### 4.4 ✏️ Параметризовать сборку сетей
- `ml/src/Models/Create.ts`: оставить как есть, но добавить generic-обёртки в `ppo/src/models/createNetworks.ts`:
  ```ts
  createPolicyNetworkGeneric({ buildBackbone, actionHeadDims, lr }): tf.LayersModel
  createValueNetworkGeneric({ buildBackbone, lr }): tf.LayersModel
  ```
- `Create.ts` теперь просто параметризует generic-функции танковой `createNetwork` из `Networks/v13`.

### 4.5 ✏️ Параметризовать CONFIG в core
- 🆕 `ppo/src/config.ts`: интерфейс `PpoConfig`:
  ```
  gamma(it), clipNorm, policyClipRatio, policyEpochs(it),
  valueClipRatio, valueEpochs(it), valueLossCoeff, valueLRCoeff,
  lrConfig, batchSize(it), miniBatchSize(it),
  adaptiveEntropy, backpressureQueueSize, savePath
  ```
- Заменить все `import { CONFIG } from 'ml-common/config'` внутри `ppo/*` на параметр `config: PpoConfig`, прокинутый в инициализаторы (`createLearnerManager(config)`, `createLearnerAgent({ config, … })`, и т.д.).

### 4.6 ✏️ Сделать `EpisodeManager` абстрактным
Перенести `ml/src/PPO/Actor/EpisodeManager.ts` → `ppo/src/core/EpisodeManager.ts`. Внутри:
- Оставить: цикл `start()`, `runEpisode()`, `runGameLoop()`, `backpressure$`.
- Убрать импорт `createScenarioByCurriculumState`, `Pilot`, `calculateFinalReward`.
- Сделать `protected abstract`:
  ```
  beforeEpisode(): Scenario
  afterEpisode(scen: Scenario): void
  cleanupEpisode(scen: Scenario): void
  runGameTick(frame, dt, scen): boolean
  awaitAgentsSync(): Promise<unknown>
  ```
- `Scenario` остаётся как generic-интерфейс с `getVehicleEids/getTeamsCount/getSuccessRatio/gameTick/destroy/train/index`.

### 4.7 📦 Перенести остальное в `ppo/src/core/` / `ppo/src/learner/`
- `ml/src/PPO/train.ts` → `ppo/src/core/train.ts`
- `ml/src/PPO/channels.ts` → `ppo/src/core/channels.ts` (с generic `AgentSample<S>`)
- `ml/src/PPO/Learner/createLearnerManager.ts` → `ppo/src/learner/createLearnerManager.ts`
- `ml/src/PPO/Learner/createLearnerAgent.ts` → `ppo/src/learner/createLearnerAgent.ts`
- `ml/src/PPO/Learner/createPolicyLearnerAgent.ts` → `ppo/src/learner/createPolicyLearnerAgent.ts`
- `ml/src/PPO/Learner/createValueLearnerAgent.ts` → `ppo/src/learner/createValueLearnerAgent.ts`
- `ml/src/PPO/Learner/isLossDangerous.ts` → `ppo/src/learner/isLossDangerous.ts`

### 4.8 🆕 Бутстрап-функции
- `ppo/src/actor/startActor.ts`:
  ```ts
  startActor({ config, backend, EpisodeManagerCtor })
  ```
- `ppo/src/learner/startPolicyLearner.ts`, `startValueLearner.ts` — то же.
Внутри: `initTensorFlow(backend)`, создать `createLearnerManager(config)` / `createPolicyLearnerAgent(...)`.

---

## Фаза 5 — Перенос домена в `ppo_tanks`

### 5.1 📦 `ml-common/consts.ts` → `ppo_tanks/src/consts.ts`
### 5.2 📦 `ml-common/config.ts` → `ppo_tanks/src/config.ts`
В нём собирается полный `CONFIG: PpoConfig & TankConfig` (домен: `episodeFrames`, `workerCount`, `savePath`, `backpressureQueueSize`).
### 5.3 📦 `ml-common/InputArrays.ts` → `ppo_tanks/src/state/InputArrays.ts`
### 5.4 📦 `ml-common/InputTensors.ts` → `ppo_tanks/src/state/InputTensors.ts`
### 5.5 📦 `ml-common/applyActionToTank.ts` → `ppo_tanks/src/state/applyActionToTank.ts`
### 5.6 📦 `ml-common/computeObstacleGrid.ts` → `ppo_tanks/src/state/computeObstacleGrid.ts`
### 5.7 📦 `ml-common/computeConnectivityMap.ts` → `ppo_tanks/src/state/computeConnectivityMap.ts`
### 5.8 📦 `ml-common/computeAllPairsDistances.ts` → `ppo_tanks/src/state/computeAllPairsDistances.ts`
### 5.9 🆕 `ppo_tanks/src/state/bindings.ts`
Экспорт `StateBindings<InputArrays>` из `createInputTensors` + `prepareRandomInputArrays`.

### 5.10 📦 `ml-common/Curriculum/**` → `ppo_tanks/src/curriculum/**`
Один шаг — папка целиком (внутренние пути не меняются), только внешние импорты.

### 5.11 📦 `ml/src/Models/Inputs.ts` → `ppo_tanks/src/models/Inputs.ts`
### 5.12 📦 `ml/src/Models/Networks/**` → `ppo_tanks/src/models/Networks/**`
### 5.13 📦 `ml/src/Models/Create.ts` → `ppo_tanks/src/models/createTankNetworks.ts`
Импортирует `createPolicyNetworkGeneric` / `createValueNetworkGeneric` из `ppo`, передаёт танковый `createNetwork` из `Networks/v13`.

### 5.14 📦 `ml/src/Reward/calculateReward.ts` → `ppo_tanks/src/reward/calculateReward.ts`

### 5.15 📦 `ml/src/PPO/Actor/EpisodeManager.ts` → `ppo_tanks/src/agents/TankEpisodeManager.ts`
Превращается в `extends EpisodeManager` из `ppo`. Реализует абстрактные хуки через `createScenarioByCurriculumState`, `Pilot`, `calculateFinalReward`.

### 5.16 📦 `ml/src/PPO/VisTest/VisTestEpisodeManager.ts` → `ppo_tanks/src/agents/VisTestEpisodeManager.ts`
Теперь `extends TankEpisodeManager`.

### 5.17 📦 `ml-common/debug.ts` → `ppo_tanks/src/ui/debug.ts`
### 5.18 📦 `ml-common/uiUtils.ts` → `ppo_tanks/src/ui/uiUtils.ts`
### 5.19 📦 `ml-common/Metrics/Browser/**` → `ppo_tanks/src/ui/MetricsBrowser/**`

---

## Фаза 6 — Точки входа в `ppo_tanks`

### 6.1 📦 `ml/index.html` → `ppo_tanks/index.html`
### 6.2 📦 `ml/config.vite.ts` → `ppo_tanks/config.vite.ts`
### 6.3 📦 `ml/test.html`, `ml/test.ts` → `ppo_tanks/`
### 6.4 ✏️ Обновить `ppo_tanks/package.json`
Скопировать `devDependencies` из старого `ml/package.json`, скрипты `dev`/`build`/`preview`.

### 6.5 🆕 `ppo_tanks/src/entry/main.ts` (бывший `ml/src/PPO/index.ts`)
```ts
import { CONFIG } from '../config';
import { bindings } from '../state/bindings';
import { VisTestEpisodeManager } from '../agents/VisTestEpisodeManager';
import { startVisTest } from 'ppo/src/...';  // или собрать тут
```

### 6.6 🆕 `ppo_tanks/src/entry/ActorWorker.ts`
Тонкий обёрточный воркер: `startActor({ config: CONFIG, backend: 'wasm', EpisodeManagerCtor: TankEpisodeManager })`.

### 6.7 🆕 `ppo_tanks/src/entry/LearnerPolicyWorker.ts`, `LearnerValueWorker.ts`
Аналогично, инжектят `createTankNetworks`, `bindings`, `CONFIG`.

### 6.8 ✏️ `ppo_tanks/index.html` указывает на `src/entry/main.ts`.

---

## Фаза 7 — Чистка

### 7.1 ✏️ Все импорты `ml-common/*` и `ml/src/*` в репо переведены на `ppo`/`ppo_tanks`
Grep-проверка: `rtk grep -rn "ml-common\|ml/src" packages/ --include="*.ts"` → 0 совпадений.

### 7.2 🗑️ Удалить `packages/ml/`
### 7.3 🗑️ Удалить `packages/ml-common/`
### 7.4 ✏️ Корневой `tsconfig.json` / `package.json`: убрать `ml`/`ml-common` paths.

---

## Фаза 8 — Верификация

### 8.1 `tsc --noEmit` по корню — зелёный.
### 8.2 `pnpm --filter ppo_tanks dev` (или yarn/npm-аналог) поднимается без ошибок в консоли.
### 8.3 Прогнать один эпизод обучения, сравнить с baseline на ветке `main`:
  - `KL`, `entropy`, `returns.std`, `explainedVariance` — в одном порядке величин.
### 8.4 Save→reload модели (IndexedDB roundtrip через `Models/Transfer.ts`).

---

## Чек-лист атомарности

Каждый шаг считается готовым, если:
- [ ] `tsc --noEmit` зелёный.
- [ ] Импорты обновлены во **всех** местах, где старый путь использовался (grep по репо).
- [ ] Нет дублирования: после move старого файла нет; если временно нужен re-export — он отмечен `// TODO: remove after step X.Y`.
- [ ] Один шаг = один коммит.

## Контрольные точки (где можно поставить на паузу)

- После **Фазы 1**: цикл разорван, `ml` и `ml-common` ещё на месте, всё работает.
- После **Фазы 2**: вся generic RL-инфра в `ppo`, но train/models ещё в `ml`.
- После **Фазы 4**: core декаплирован, но физически файлы ещё могут лежать в `ml/src/PPO`.
- После **Фазы 5**: домен в `ppo_tanks`, остались точки входа.
- После **Фазы 7**: старые пакеты удалены.
