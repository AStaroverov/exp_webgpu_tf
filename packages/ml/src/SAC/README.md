# SAC Quick Start Guide

## Запуск обучения

```bash
cd packages/ml
npm run dev
```

Откройте в браузере: `http://localhost:5173/sac.html`

## Структура проекта SAC

```
packages/ml/src/SAC/
├── index.ts                  # Главная точка входа
├── train.ts                  # Функции обучения (actor, critics, temperature)
├── channels.ts               # Каналы связи между workers
├── ActorWorker.ts            # Worker для сбора данных
├── Actor/
│   └── EpisodeManager.ts     # Управление эпизодами
├── Learner/
│   ├── LearnerActorWorker.ts      # Worker для обучения actor
│   ├── LearnerCriticWorker.ts     # Worker для обучения critics
│   ├── createLearnerManager.ts    # Координация обучения
│   ├── createActorLearner.ts      # Learner для actor
│   ├── createCriticLearner.ts     # Learner для critics
│   ├── createLearnerAgent.ts      # Базовый learner agent
│   └── isLossDangerous.ts         # Валидация loss
└── VisTest/
    └── VisTestEpisodeManager.ts   # Визуализация

packages/ml-common/
├── config.ts                 # SAC_CONFIG - гиперпараметры
├── Memory.ts                 # SACMemory и SACMemoryBatch
├── SACReplayBuffer.ts        # Replay buffer с PER
└── computeLogProb.ts         # sampleActionWithTanhSquashing

packages/ml/src/Models/
├── Create.ts                 # createCriticNetwork
├── def.ts                    # Model enum (Critic1, Critic2, etc.)
└── Utils.ts                  # softUpdateTargetNetwork
```

## Архитектура SAC

### Компоненты

1. **Actor (Policy)** - π(a|s)
   - Генерирует действия с tanh squashing
   - Обучается максимизировать Q(s,a) - α*log π(a|s)

2. **Critic Networks** - Q1(s,a), Q2(s,a)
   - Twin Q-networks для уменьшения overestimation
   - Обучаются по Bellman backup с entropy регуляризацией

3. **Target Networks** - Q1_target, Q2_target
   - Стабилизируют обучение
   - Обновляются через Polyak averaging (τ=0.005)

4. **Temperature α**
   - Entropy coefficient
   - Балансирует exploration/exploitation
   - Опционально: auto-tuning

5. **Replay Buffer**
   - Хранит (s, a, r, s', done)
   - Off-policy learning
   - Опционально: PER (Prioritized Experience Replay)

### Workers

- **ActorWorker** (x4) - собирает данные параллельно
- **LearnerActorWorker** - обучает policy
- **LearnerCriticWorker** - обучает оба Q-networks

### Training Loop

1. Actors собирают данные → Replay Buffer
2. Learners сэмплируют батчи из Replay Buffer
3. Train Critics: minimize Bellman error
4. Train Actor: maximize Q(s,a) - α*log π(a|s)
5. Soft update targets: θ_target = τ*θ + (1-τ)*θ_target

## Гиперпараметры (SAC_CONFIG)

```typescript
{
  alpha: 0.2,              // Temperature
  tau: 0.005,              // Soft update rate
  gamma: 0.99,             // Discount factor
  replayBufferSize: 1M,    // Buffer size
  batchSize: 256,          // Mini-batch size
  actorLR: 3e-4,           // Actor learning rate
  criticLR: 3e-4,          // Critic learning rate
  clipNorm: 1.0,           // Gradient clipping
  minLogStd: -20,          // Min log std
  maxLogStd: 2,            // Max log std
}
```

## Метрики

### Actor
- Actor loss
- Policy entropy
- Action statistics (mean/std)

### Critics
- Critic1 loss, Critic2 loss
- Q1 values, Q2 values
- Target Q values
- TD error

### Replay Buffer
- Buffer size
- Sampling statistics

### Episode
- Reward, Length
- Win rate, Survival time

## Следующие шаги

1. **Тестирование** (Фаза 8):
   - Проверить сходимость на простой задаче
   - Отладить NaN/Inf если возникнут
   - Проверить memory usage

2. **Оптимизация** (Фаза 9):
   - Настройка гиперпараметров
   - Профилирование производительности
   - Добавить SAC-специфичные UI метрики

3. **Документация** (Фаза 10):
   - A/B тестирование PPO vs SAC
   - Анализ результатов

## TODO: Agent Integration

Для полной функциональности требуется обновить Agent классы.
См. `packages/ml/src/SAC/TODO_AGENT_INTEGRATION.md`

## Отличия от PPO

| Аспект | PPO | SAC |
|--------|-----|-----|
| Learning | On-policy | Off-policy |
| Buffer | Episodic | Replay buffer |
| Value | V(s) | Q(s,a) twin networks |
| Updates | Clipped ratio | Entropy maximization |
| Exploration | Stochastic policy | Maximum entropy |
| Data efficiency | Low | High |
| Stability | Good | Better |

## Troubleshooting

### NaN/Inf в loss
- Проверить bounds для log_std
- Увеличить gradient clipping
- Проверить numerical stability в tanh correction

### Медленное обучение
- Увеличить learning rates
- Проверить размер replay buffer
- Увеличить частоту обновлений

### Memory leaks
- Проверить disposal tensors
- Мониторить `tf.memory()`
- Использовать `tf.tidy()`

---

**Версия:** 1.0  
**Дата:** 17 октября 2025
