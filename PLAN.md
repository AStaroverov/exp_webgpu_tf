# План перевода с PPO на SAC (Soft Actor-Critic)

## Обзор изменений

Переход от PPO (Proximal Policy Optimization) к SAC (Soft Actor-Critic) для улучшения стабильности обучения и эффективности использования данных в multi-agent окружении с танками.

---

## 1. Архитектурные изменения сетей

### 1.1. Policy Network (Actor)

**Текущая реализация (PPO):**
- Выходы: `mean` и `log_std` для Gaussian policy
- Сэмплирование с клиппингом ratio

**Изменения для SAC:**
- ✅ Оставить выходы `mean` и `log_std`
- ➕ Добавить reparameterization trick: `action = mean + std * noise`
- ➕ Применить `tanh` squashing для ограничения действий в [-1, 1]
- ➕ Вычислять log-probability с коррекцией на tanh:
  ```
  log_π(a|s) = log_π(u|s) - Σ log(1 - tanh²(u))
  ```
- 🗑️ Убрать клиппинг PPO (clip ratio)

**Файлы для изменения:**
- `packages/ml/src/Models/Create.ts` - модифицировать `createPolicyNetwork()`

### 1.2. Critic Networks (Q-functions)

**Текущая реализация (PPO):**
- Одна Value Network: `V(s)` - оценивает только состояние

**Изменения для SAC:**
- 🗑️ Удалить `createValueNetwork()`
- ➕ Создать **две Q-Networks**: `Q1(s,a)` и `Q2(s,a)` (twin Q-networks для уменьшения overestimation bias)
- ➕ Создать **две Target Q-Networks**: `Q1_target` и `Q2_target` (для стабильности обучения)
- ➕ Входы: state + action (конкатенация после encoding состояния)
- ➕ Выход: скалярное значение Q(s,a)

**Файлы для создания:**
- `packages/ml/src/Models/Create.ts` - добавить `createCriticNetwork()`

### 1.3. Temperature Parameter (α)

- ➕ Добавить learnable температурный параметр `α` (entropy coefficient)
- ➕ Опционально: автоматическая подстройка α для поддержания целевой энтропии
- ➕ Целевая энтропия: `-dim(action_space)` = `-ACTION_DIM`

**Файлы для изменения:**
- `packages/ml/src/SAC/train.ts` - добавить `trainTemperature()`

---

## 2. Изменения в Replay Buffer

### 2.1. Структура данных

**Текущая реализация (PPO):**
```typescript
{
  states,        // состояния
  actions,       // действия
  rewards,       // награды
  oldLogProbs,   // старые log-probabilities
  oldValues,     // старые value estimates
  advantages,    // преимущества (GAE)
  returns        // дисконтированные returns (V-trace)
}
```

**Для SAC:**
```typescript
{
  states,        // текущее состояние s_t
  actions,       // выбранное действие a_t
  rewards,       // полученная награда r_t
  next_states,   // следующее состояние s_{t+1}
  dones          // флаг окончания эпизода
}
```

### 2.2. Функционал

- 🗑️ Убрать вычисление `advantages` и `returns` (GAE/V-trace не нужны для SAC)
- 🗑️ Убрать сохранение `oldLogProbs`, `oldValues`
- ➕ Добавить хранение `next_states` и `dones`
- ✅ Использовать существующий `ReplayBuffer` с uniform sampling
- ➕ **Рекомендуется**: переключиться на `PrioritizedReplayBuffer` (уже реализован в `packages/ml-common/PrioritizedReplayBuffer.ts`)

**Файлы для изменения:**
- `packages/ml-common/Memory.ts` - обновить структуру `AgentMemoryBatch`
- `packages/ml/src/SAC/Actor/EpisodeManager.ts` - изменить логику сбора данных

---

## 3. Изменения в обучении

### 3.1. Файловая структура

**Создать новую структуру:**
```
packages/ml/src/
  SAC/                              # новая папка (вместо PPO/)
    index.ts                        # точка входа
    train.ts                        # функции обучения
    ActorWorker.ts                  # worker для сбора данных
    LearnerActorWorker.ts           # worker для обучения policy
    LearnerCriticWorker.ts          # worker для обучения Q-functions
    channels.ts                     # каналы связи между workers
    Actor/
      EpisodeManager.ts             # управление эпизодами и сбором данных
    Learner/
      createLearnerManager.ts       # координация обучения
      createActorLearner.ts         # обучение actor (policy)
      createCriticLearner.ts        # обучение critics (Q-functions)
      isLossDangerous.ts            # проверка валидности loss
    VisTest/
      VisTestEpisodeManager.ts      # визуализация и тестирование
```

### 3.2. Training Loop - Actor (Policy)

**Удалить:**
```typescript
trainPolicyNetwork(
  network, states, actions, oldLogProbs, 
  advantages, clipRatio, entropyCoeff
)
```

**Добавить:**
```typescript
trainActorNetwork(
  actorNetwork: tf.LayersModel,
  criticNetwork1: tf.LayersModel,
  criticNetwork2: tf.LayersModel,
  states: tf.Tensor[],
  alpha: number | tf.Variable,  // temperature parameter
  batchSize: number,
  clipNorm: number,
  minLogStd: number,
  maxLogStd: number,
  returnCost: boolean,
): tf.Tensor | undefined {
  return tf.tidy(() => {
    return optimize(actorNetwork.optimizer, () => {
      // 1. Sample actions from current policy with reparameterization
      const { action, logProb } = sampleActionWithReparam(
        actorNetwork, states, minLogStd, maxLogStd
      );
      
      // 2. Compute Q-values from both critics
      const q1 = criticNetwork1.predict([...states, action], { batchSize });
      const q2 = criticNetwork2.predict([...states, action], { batchSize });
      const minQ = tf.minimum(q1, q2).squeeze();
      
      // 3. Actor loss: E[α * log π(a|s) - Q(s,a)]
      // Maximize Q - entropy bonus
      const alphaValue = typeof alpha === 'number' ? alpha : alpha.read();
      const actorLoss = tf.scalar(alphaValue).mul(logProb).sub(minQ).mean();
      
      return actorLoss as tf.Scalar;
    }, { clipNorm, returnCost });
  });
}
```

### 3.3. Training Loop - Critic (Q-functions)

**Удалить:**
```typescript
trainValueNetwork(
  network, states, returns, oldValues, clipRatio
)
```

**Добавить:**
```typescript
trainCriticNetworks(
  critic1: tf.LayersModel,
  critic2: tf.LayersModel,
  targetCritic1: tf.LayersModel,
  targetCritic2: tf.LayersModel,
  actorNetwork: tf.LayersModel,
  batch: {
    states: tf.Tensor[],
    actions: tf.Tensor,
    rewards: tf.Tensor,
    nextStates: tf.Tensor[],
    dones: tf.Tensor
  },
  alpha: number | tf.Variable,
  gamma: number,
  batchSize: number,
  clipNorm: number,
  minLogStd: number,
  maxLogStd: number,
  returnCost: boolean,
): { loss1: tf.Tensor | undefined, loss2: tf.Tensor | undefined } {
  return tf.tidy(() => {
    // 1. Sample next actions from current policy
    const { action: nextAction, logProb: nextLogProb } = 
      sampleActionWithReparam(
        actorNetwork, batch.nextStates, minLogStd, maxLogStd
      );
    
    // 2. Compute target Q-value with entropy regularization
    const targetQ1 = targetCritic1.predict(
      [...batch.nextStates, nextAction], { batchSize }
    ).squeeze();
    const targetQ2 = targetCritic2.predict(
      [...batch.nextStates, nextAction], { batchSize }
    ).squeeze();
    const minTargetQ = tf.minimum(targetQ1, targetQ2);
    
    const alphaValue = typeof alpha === 'number' ? alpha : alpha.read();
    const target = batch.rewards.add(
      tf.scalar(gamma).mul(
        tf.scalar(1).sub(batch.dones).mul(
          minTargetQ.sub(tf.scalar(alphaValue).mul(nextLogProb))
        )
      )
    );
    const targetDetached = target.stopGradient();
    
    // 3. Compute current Q-values
    const currentQ1 = critic1.predict(
      [...batch.states, batch.actions], { batchSize }
    ).squeeze();
    const currentQ2 = critic2.predict(
      [...batch.states, batch.actions], { batchSize }
    ).squeeze();
    
    // 4. MSE loss for both critics
    const loss1 = optimize(critic1.optimizer, () => {
      return tf.losses.meanSquaredError(targetDetached, currentQ1)
        .mean() as tf.Scalar;
    }, { clipNorm, returnCost });
    
    const loss2 = optimize(critic2.optimizer, () => {
      return tf.losses.meanSquaredError(targetDetached, currentQ2)
        .mean() as tf.Scalar;
    }, { clipNorm, returnCost });
    
    return { loss1, loss2 };
  });
}
```

### 3.4. Temperature (α) Update

```typescript
trainTemperature(
  alpha: tf.Variable,
  logProbs: tf.Tensor,
  targetEntropy: number,  // обычно -dim(action_space) = -ACTION_DIM
  learningRate: number,
  clipNorm: number,
  returnCost: boolean,
): tf.Tensor | undefined {
  return tf.tidy(() => {
    // Alpha loss: -α * (log π + target_entropy)
    // Minimize: maximize entropy when it's below target
    return optimize(
      new tf.train.adam(learningRate), 
      () => {
        const alphaLoss = alpha.mul(
          logProbs.add(tf.scalar(targetEntropy))
        ).mean().mul(-1);
        return alphaLoss as tf.Scalar;
      },
      { clipNorm, returnCost }
    );
  });
}
```

### 3.5. Soft Target Update

```typescript
softUpdateTargetNetworks(
  critic1: tf.LayersModel,
  critic2: tf.LayersModel,
  targetCritic1: tf.LayersModel,
  targetCritic2: tf.LayersModel,
  tau: number = 0.005,  // soft update coefficient
): void {
  // Polyak averaging: θ_target = τ * θ + (1 - τ) * θ_target
  updateWeightsSoft(critic1, targetCritic1, tau);
  updateWeightsSoft(critic2, targetCritic2, tau);
}

function updateWeightsSoft(
  source: tf.LayersModel,
  target: tf.LayersModel,
  tau: number,
): void {
  const sourceWeights = source.getWeights();
  const targetWeights = target.getWeights();
  
  const updatedWeights = targetWeights.map((targetWeight, i) => {
    return tf.tidy(() => {
      const sourceWeight = sourceWeights[i];
      return sourceWeight.mul(tau).add(
        targetWeight.mul(1 - tau)
      );
    });
  });
  
  target.setWeights(updatedWeights);
  
  // Dispose old weights
  targetWeights.forEach(w => w.dispose());
}
```

**Файлы для создания/изменения:**
- `packages/ml/src/SAC/train.ts` - все функции обучения
- `packages/ml/src/Models/Utils.ts` - добавить `updateWeightsSoft()`

---

## 4. Actor (сбор данных)

### 4.1. EpisodeManager

**Изменения:**
- 🗑️ Убрать вычисление GAE (Generalized Advantage Estimation)
- 🗑️ Убрать вычисление V-trace
- 🗑️ Убрать сохранение `oldValues` и `oldLogProbs`
- ➕ Сохранять переходы в формате `(s, a, r, s', done)` вместо `(s, a, r, value, logProb)`
- ➕ Добавить логику сохранения `next_state`
- ➕ Добавить флаг `done` для терминальных состояний
- ➕ Опционально: добавить exploration noise на начальных этапах (Ornstein-Uhlenbeck или Gaussian)

**Файлы для изменения:**
- `packages/ml/src/SAC/Actor/EpisodeManager.ts`
- `packages/ml-common/Memory.ts`

### 4.2. Inference

- ✅ Оставить стохастический выбор действий во время сбора данных
- ➕ Применять `tanh` squashing для ограничения действий
- ➕ Для evaluation/testing: использовать `mean` без шума (детерминированная политика)
- ➕ Reparameterization trick для backprop через сэмплирование

**Файлы для изменения:**
- `packages/ml/src/SAC/Actor/EpisodeManager.ts`
- `packages/ml-common/computeLogProb.ts` - добавить функцию с tanh correction

---

## 5. Новые модели

### 5.1. Create.ts - Critic Network

```typescript
export function createCriticNetwork(): tf.LayersModel {
  // State inputs (как в policy network)
  const {
    controllerInput,
    battleInput,
    tankInput,
    enemiesInput,
    alliesInput,
    bulletsInput,
  } = createInputs();
  
  // Action input
  const actionInput = tf.input({
    shape: [ACTION_DIM],
    name: 'action_input',
    dtype: 'float32',
  });
  
  // Encode states через transformer (переиспользуем логику из policy)
  const tokens = convertInputsToTokens(
    Model.Critic1, // или Critic2
    controllerInput,
    battleInput,
    tankInput,
    enemiesInput,
    alliesInput,
    bulletsInput,
    criticNetworkConfig.dim,
  );
  
  const transformed = applySelfTransformLayers(
    Model.Critic1,
    tokens,
    criticNetworkConfig.dim,
    criticNetworkConfig.heads,
    criticNetworkConfig.dropout,
  );
  
  // Global pooling для state representation
  const stateEncoding = tf.layers.globalAveragePooling1d({
    name: Model.Critic1 + '_pool',
  }).apply(transformed) as tf.SymbolicTensor;
  
  // Concatenate state encoding + action
  const combined = tf.layers.concatenate({
    name: Model.Critic1 + '_concat',
  }).apply([stateEncoding, actionInput]) as tf.SymbolicTensor;
  
  // MLP layers
  const hidden = applyDenseLayers(
    Model.Critic1 + '_mlp',
    combined,
    criticNetworkConfig.finalMLP,
  );
  
  // Q-value output (single scalar)
  const qValue = tf.layers.dense({
    name: Model.Critic1 + '_q_value',
    units: 1,
    activation: 'linear',
  }).apply(hidden) as tf.SymbolicTensor;
  
  const model = tf.model({
    inputs: [
      controllerInput,
      battleInput,
      tankInput,
      enemiesInput,
      alliesInput,
      bulletsInput,
      actionInput,
    ],
    outputs: qValue,
    name: Model.Critic1,
  });
  
  model.compile({
    optimizer: new PatchedAdamOptimizer(CONFIG.lr),
    loss: 'meanSquaredError',
  });
  
  return model;
}
```

### 5.2. Model Enum

```typescript
// packages/ml/src/Models/def.ts
export enum Model {
  Policy = 'policy',        // или переименовать в Actor
  Critic1 = 'critic1',
  Critic2 = 'critic2',
  TargetCritic1 = 'target_critic1',
  TargetCritic2 = 'target_critic2',
}
```

### 5.3. Network Config

```typescript
// packages/ml/src/Models/Create.ts
const criticNetworkConfig: NetworkConfig = {
  dim: 64,               // размер embeddings
  heads: 4,              // количество attention heads
  dropout: 0.0,          // dropout rate
  finalMLP: [
    ['relu', 256],
    ['relu', 256],
    ['relu', 128],
  ] as [ActivationIdentifier, number][],
};
```

**Файлы для изменения:**
- `packages/ml/src/Models/Create.ts`
- `packages/ml/src/Models/def.ts`

---

## 6. Гиперпараметры

### 6.1. Удалить (PPO-специфичные)

- ❌ `clipRatio` (0.2) - PPO clipping
- ❌ `entropyCoeff` (0.01) - заменить на `alpha`
- ❌ `vfCoeff` (0.5) - value function coefficient
- ❌ GAE параметры (`lambda`, `gamma` для GAE)

### 6.2. Добавить (SAC-специфичные)

```typescript
// packages/ml-common/config.ts
export const SAC_CONFIG = {
  // Temperature
  alpha: 0.2,                    // entropy coefficient (или auto-tune)
  autoTuneAlpha: true,           // автоматическая подстройка
  targetEntropy: -ACTION_DIM,    // целевая энтропия для auto-tune
  alphaLR: 3e-4,                 // learning rate для alpha
  
  // Soft target update
  tau: 0.005,                    // коэффициент Polyak averaging
  
  // Discount factor
  gamma: 0.99,                   // discount factor
  
  // Replay buffer
  replayBufferSize: 1_000_000,   // размер replay buffer
  prioritizedReplay: true,       // использовать PER
  priorityAlpha: 0.6,            // приоритизация
  priorityBeta: 0.4,             // importance sampling
  
  // Training
  batchSize: 256,                // размер batch
  actorUpdateFreq: 1,            // частота обновления actor
  criticUpdateFreq: 1,           // частота обновления critics
  targetUpdateFreq: 1,           // частота обновления target networks
  
  // Learning rates
  actorLR: 3e-4,                 // learning rate для actor
  criticLR: 3e-4,                // learning rate для critics
  
  // Gradient clipping
  clipNorm: 1.0,                 // gradient clipping norm
  
  // Exploration
  initialRandomSteps: 10000,     // начальные random шаги
  
  // Log std bounds
  minLogStd: -20,                // минимальное log std
  maxLogStd: 2,                  // максимальное log std
};
```

**Файлы для создания/изменения:**
- `packages/ml-common/config.ts` - добавить `SAC_CONFIG`

---

## 7. Пошаговый план миграции

### Фаза 1: Подготовка (1-2 дня)

**Задачи:**
1. ✅ Изучить текущую архитектуру PPO
2. ✅ Создать ветку `feat/sac`
3. ✅ Создать папку `packages/ml/src/SAC/`
4. ✅ Скопировать базовую структуру из `PPO/`
5. ✅ Создать `SAC_CONFIG` в `packages/ml-common/config.ts`

**Критерий готовности:**
- ✅ Структура папок создана
- ✅ Базовые файлы скопированы
- ✅ Конфигурация определена

### Фаза 2: Модели (2-3 дня)

**Задачи:**
1. ✅ Создать `createCriticNetwork()` в `packages/ml/src/Models/Create.ts`
2. ✅ Добавить `Critic1`, `Critic2`, `TargetCritic1`, `TargetCritic2` в `Model` enum
3. ✅ Реализовать `softUpdateTargetNetwork()` для soft target update
4. ✅ Добавить `sampleActionWithTanhSquashing()` с tanh squashing и log-prob correction
5. ⬜ Создать функцию `sampleActionWithReparam()` с reparameterization trick
6. ⬜ Протестировать forward pass всех сетей
7. ⬜ Проверить размерности входов/выходов

**Критерий готовности:**
- ⚠️ Частично: Все 4 сети (Actor, Critic1, Critic2, Target Critics) создаются без ошибок
- ⬜ Forward pass работает корректно
- ✅ Soft update реализован

**Файлы:**
- ✅ `packages/ml/src/Models/Create.ts`
- ✅ `packages/ml/src/Models/def.ts`
- ✅ `packages/ml/src/Models/Utils.ts`
- ✅ `packages/ml-common/computeLogProb.ts`

### Фаза 3: Training (3-4 дня)

**Задачи:**
1. ✅ Создать `packages/ml/src/SAC/train.ts`
2. ✅ Реализовать `trainCriticNetworks()`
3. ✅ Реализовать `trainActorNetwork()`
4. ✅ Реализовать `trainTemperature()` для auto-tuning alpha
5. ✅ Реализовать helper функции (`parsePredict`, `optimize`)
6. ✅ Реализовать `act()` для inference с deterministic режимом
7. ⬜ Убрать функции PPO: `trainPolicyNetwork()`, `trainValueNetwork()` (оставим для совместимости)
8. ⬜ Убрать V-trace/GAE: `computeVTrace()`, `computeGAE()` (не нужно трогать PPO)
9. ⬜ Добавить тесты для функций обучения

**Критерий готовности:**
- ✅ Все функции обучения реализованы
- ⬜ Loss значения в разумных пределах (протестируем в Фазе 8)
- ⬜ Градиенты не становятся NaN/Inf (протестируем в Фазе 8)

**Файлы:**
- ✅ `packages/ml/src/SAC/train.ts`

### Фаза 4: Replay Buffer (1-2 дня)

**Задачи:**
1. ✅ Обновить `AgentMemoryBatch` в `packages/ml-common/Memory.ts`:
   - ✅ Создать новый тип `SACMemoryBatch`
   - ✅ Добавить `nextStates: InputArrays[]`
   - ✅ Сохранить `dones: Float32Array`
2. ✅ Создать класс `SACMemory` для сбора данных
3. ✅ Создать `SACReplayBuffer` для управления replay buffer
4. ✅ Настроить поддержку `PrioritizedReplayBuffer` (готово к использованию)
5. ✅ Добавить методы для добавления батчей и сэмплирования

**Критерий готовности:**
- ✅ Новая структура данных `SACMemoryBatch` создана
- ✅ Класс `SACMemory` реализован
- ✅ `SACReplayBuffer` готов к использованию
- ✅ Поддержка PER добавлена

**Файлы:**
- ✅ `packages/ml-common/Memory.ts`
- ✅ `packages/ml-common/SACReplayBuffer.ts`

### Фаза 5: Actor Workers (2-3 дня)

**Задачи:**
1. ✅ Создать `packages/ml/src/SAC/Actor/EpisodeManager.ts`
2. ✅ Обновить логику сбора данных:
   - ✅ Использовать `SACMemory` для сбора (s, a, r, s', done)
   - ✅ Убрать вызовы value network (не нужны в SAC)
   - ✅ Убрать вычисление GAE/V-trace (не нужны в SAC)
3. ⬜ Добавить логику определения `done` флага (уже есть в базовом коде)
4. ⬜ Добавить сохранение `next_state` (требуется обновление Agent)
5. ✅ Обновить `packages/ml/src/SAC/ActorWorker.ts`
6. ⬜ Добавить initial random exploration (опционально, для Фазы 9)

**Критерий готовности:**
- ✅ EpisodeManager создан и адаптирован для SAC
- ⬜ Actor корректно собирает данные в новом формате (требует обновления Agent)
- ⬜ Переходы сохраняются с `next_states` и `dones` (требует обновления Agent)
- ✅ Нет memory leaks

**Файлы:**
- ✅ `packages/ml/src/SAC/Actor/EpisodeManager.ts`
- ✅ `packages/ml/src/SAC/ActorWorker.ts`
- ✅ `packages/ml/src/SAC/TODO_AGENT_INTEGRATION.md` (документация для следующих шагов)

**Примечание:**
Для полной интеграции потребуется обновить Agent классы (CurrentActorAgent, HistoricalAgent).
TODO файл создан с детальными инструкциями.

### Фаза 6: Learner Workers (3-4 дня) ✅ ЗАВЕРШЕНО

**Задачи:**
1. ✅ Создать `packages/ml/src/SAC/Learner/createActorLearner.ts`
2. ✅ Создать `packages/ml/src/SAC/Learner/createCriticLearner.ts`
3. ✅ Обновить `packages/ml/src/SAC/Learner/createLearnerManager.ts`:
   - Координация обучения actor и critics
   - Soft target updates
   - Temperature updates (если auto-tune)
4. ✅ Создать `packages/ml/src/SAC/Learner/LearnerActorWorker.ts`
5. ✅ Создать `packages/ml/src/SAC/Learner/LearnerCriticWorker.ts`
6. ✅ Настроить каналы связи между workers в `packages/ml/src/SAC/channels.ts`
7. ✅ Добавить логику загрузки/сохранения всех 4 моделей

**Критерий готовности:**
- ✅ Workers корректно обучают свои модели
- ✅ Soft updates работают
- ✅ Модели сохраняются и загружаются
- ✅ Нет race conditions

**Файлы:**
- ✅ `packages/ml/src/SAC/Learner/createActorLearner.ts` - обучает actor через минимум twin Q-values
- ✅ `packages/ml/src/SAC/Learner/createCriticLearner.ts` - обучает оба critic networks с soft updates
- ✅ `packages/ml/src/SAC/Learner/createLearnerManager.ts` - координирует replay buffer и learners
- ✅ `packages/ml/src/SAC/Learner/LearnerActorWorker.ts` - worker процесс для actor
- ✅ `packages/ml/src/SAC/Learner/LearnerCriticWorker.ts` - worker процесс для обоих critics
- ✅ `packages/ml/src/SAC/Learner/createLearnerAgent.ts` - обертка для learner agents
- ✅ `packages/ml/src/SAC/Learner/isLossDangerous.ts` - валидация loss значений

**Примечание:**
Temperature auto-tuning реализован в `trainTemperature()` но пока не подключен к learner worker (опциональная фича).

### Фаза 7: Интеграция и точка входа (1 день) ✅ ЗАВЕРШЕНО

**Задачи:**
1. ✅ Обновить `packages/ml/src/SAC/index.ts` (точка входа)
2. ✅ Создать `packages/ml/sac.html` (копия `appo.html`)
3. ✅ Создать `packages/ml/sac.ts` (копия `appo.ts`)
4. ✅ Обновить `packages/ml/config.vite.ts` для SAC entry point (уже готов)
5. ✅ Создать `packages/ml/src/SAC/VisTest/VisTestEpisodeManager.ts` для визуализации

**Критерий готовности:**
- ✅ SAC запускается через `sac.html`
- ✅ Workers создаются корректно
- ⚠️ UI отображает метрики (базовые метрики работают, SAC-специфичные в Фазе 9)

**Файлы:**
- ✅ `packages/ml/src/SAC/index.ts` - инициализирует TensorFlow, создает actor и learner workers
- ✅ `packages/ml/sac.html` - HTML точка входа для SAC
- ✅ `packages/ml/sac.ts` - импортирует SAC/index.ts
- ✅ `packages/ml/src/SAC/VisTest/VisTestEpisodeManager.ts` - менеджер визуализации
- ✅ `packages/ml/config.vite.ts` - уже настроен (не требует изменений)

**Примечание:**
Для запуска: `npm run dev` и открыть `http://localhost:5173/sac.html`

### Фаза 8: Тестирование (2-3 дня)

**Задачи:**
1. ⬜ Создать unit-тесты для:
   - `sampleActionWithReparam()`
   - `trainActorNetwork()`
   - `trainCriticNetworks()`
   - `trainTemperature()`
   - `softUpdateTargetNetworks()`
2. ⬜ Проверить сходимость на простой задаче (например, single agent vs bot)
3. ⬜ Проверить, что Q-values не diverge
4. ⬜ Проверить, что alpha корректно подстраивается (если auto-tune)
5. ⬜ Отладка NaN/Inf в loss:
   - Проверить gradient clipping
   - Проверить bounds для log_std
   - Проверить numerical stability в log-prob calculation
6. ⬜ Проверить memory usage (4 networks могут занимать много памяти)

**Критерий готовности:**
- Все тесты проходят
- Обучение сходится на простой задаче
- Нет NaN/Inf в метриках
- Memory usage приемлемый

**Файлы:**
- `packages/ml/src/SAC/train.test.ts`
- `packages/ml/src/Models/Create.test.ts`

### Фаза 9: Оптимизация (1-2 дня)

**Задачи:**
1. ⬜ Настройка гиперпараметров:
   - Learning rates (actor, critics, alpha)
   - Batch size
   - Tau (soft update coefficient)
   - Alpha (temperature)
   - Replay buffer size
2. ⬜ Профилирование производительности:
   - Время обучения step
   - Memory footprint
   - GPU utilization
3. ⬜ Оптимизации:
   - Batch inference где возможно
   - Переиспользование tensors
   - Уменьшение memory allocations

**Критерий готовности:**
- Производительность сравнима или лучше PPO
- Гиперпараметры настроены
- Нет bottlenecks

### Фаза 10: Документация и сравнение (1-2 дня)

**Задачи:**
1. ⬜ Написать документацию:
   - Как запустить SAC
   - Описание гиперпараметров
   - Отличия от PPO
   - Troubleshooting
2. ⬜ A/B тестирование PPO vs SAC:
   - Сравнить sample efficiency
   - Сравнить финальную производительность
   - Сравнить стабильность обучения
3. ⬜ Создать графики сравнения
4. ⬜ Обновить README.md

**Критерий готовности:**
- Документация написана
- Есть сравнительный анализ
- Результаты задокументированы

**Файлы:**
- `packages/ml/README.md`
- `SAC_GUIDE.md` (новый файл)

---

## 8. Риски и митигация

### Риски

| Риск | Вероятность | Влияние | Митигация |
|------|-------------|---------|-----------|
| SAC менее sample-efficient на старте | Высокая | Среднее | Использовать PER, увеличить replay buffer |
| Требуется больше памяти (4 networks) | Высокая | Высокое | Мониторить memory, уменьшить размер networks если нужно |
| Сложнее настроить гиперпараметры | Средняя | Среднее | Использовать auto-tuning для alpha, grid search |
| NaN/Inf в обучении | Средняя | Высокое | Gradient clipping, проверить numerical stability |
| Медленнее обучение из-за 4 networks | Средняя | Среднее | Оптимизировать inference, использовать batch operations |
| Нестабильность в multi-agent среде | Средняя | Высокое | Постепенное введение сложности (curriculum learning) |

### Рекомендации

1. 🔄 **Сохранить PPO код** в отдельной ветке для возможности rollback
2. 📊 **A/B тестирование** PPO vs SAC на протяжении миграции
3. 🎯 **Auto-tuning alpha** упростит настройку entropy regularization
4. 💾 **Использовать PrioritizedReplayBuffer** для ускорения обучения
5. 📈 **Расширенный мониторинг**:
   - Q-values (Q1, Q2, min Q, target Q)
   - Alpha (temperature)
   - Policy entropy
   - Actor loss, Critic losses
   - Gradient norms
   - Replay buffer statistics
6. 🧪 **Incremental rollout**: сначала протестировать на single-agent, потом multi-agent
7. 📝 **Checkpoint часто**: сохранять модели после каждой успешной фазы

---

## 9. Потенциальные преимущества SAC

### По сравнению с PPO

- ✅ **Лучше исследует пространство действий** благодаря maximum entropy objective
- ✅ **Более стабильное обучение** благодаря off-policy learning и twin critics
- ✅ **Эффективнее использует данные** через replay buffer (каждый sample используется многократно)
- ✅ **Подходит для continuous action spaces** (что у нас и есть)
- ✅ **Автоматическая балансировка exploration/exploitation** через temperature parameter α
- ✅ **Не требует вычисления advantages** (проще и быстрее)
- ✅ **Меньше гиперпараметров для настройки** (с auto-tuning alpha)

### Для нашей задачи (multi-agent tanks)

- ✅ **Лучше справляется с sparse rewards** благодаря exploration
- ✅ **Более robustная к изменениям в окружении** (non-stationary multi-agent environment)
- ✅ **Лучше переносится на разные сценарии** благодаря maximum entropy policy

---

## 10. Метрики для мониторинга

### Обязательные метрики

1. **Actor metrics:**
   - Actor loss
   - Policy entropy (средняя энтропия действий)
   - Action mean/std statistics
   - Gradient norms

2. **Critic metrics:**
   - Critic1 loss
   - Critic2 loss
   - Q1 values (mean, min, max)
   - Q2 values (mean, min, max)
   - Target Q values
   - TD error

3. **Alpha (temperature):**
   - Current alpha value
   - Alpha loss (если auto-tune)
   - Target entropy vs actual entropy

4. **Replay buffer:**
   - Buffer size
   - Sampling statistics
   - Priority statistics (если PER)

5. **Episode metrics:**
   - Episode reward
   - Episode length
   - Win rate
   - Survival time

### Дополнительные метрики

- Q-value overestimation (Q1 vs Q2 difference)
- Action distribution visualization
- State-value estimation accuracy
- Sample efficiency (reward per timestep)

---

## 11. Чеклист перед запуском

### Pre-flight checks

- [ ] Все 4 сети создаются без ошибок
- [ ] Forward pass работает для всех сетей
- [ ] Soft target update корректно копирует веса
- [ ] Replay buffer сохраняет данные в правильном формате
- [ ] Actor собирает данные с `next_states` и `dones`
- [ ] Learners корректно обучают свои модели
- [ ] Градиенты не взрываются (NaN/Inf check)
- [ ] Memory usage в пределах нормы
- [ ] Все workers запускаются без ошибок
- [ ] Метрики логируются и отображаются
- [ ] Модели сохраняются и загружаются
- [ ] Есть baseline для сравнения с PPO

---

## 12. Следующие шаги после SAC

После успешной миграции на SAC, можно рассмотреть:

1. **TD3 (Twin Delayed DDPG)** - упрощенная версия SAC без entropy term
2. **Rainbow DQN** - для дискретных действий (если понадобится)
3. **Distributed SAC** - параллельное обучение нескольких агентов
4. **Multi-agent SAC** - явный учет других агентов в Q-функциях
5. **Hierarchical RL** - разделение на high-level и low-level политики
6. **Meta-learning** - быстрая адаптация к новым сценариям

---

**Версия:** 1.0  
**Дата:** 17 октября 2025  
**Статус:** Draft  
**Автор:** AI Assistant
