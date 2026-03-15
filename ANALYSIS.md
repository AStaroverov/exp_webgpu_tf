# Анализ проекта exp_webgpu_tf

## Обзор

Монорепозиторий (npm workspaces) для обучения RL-агентов в браузерной танковой арене. Весь стек — TypeScript, TensorFlow.js (WASM/WebGPU), Vite.

**Пакеты:**

| Пакет | Назначение |
|-------|-----------|
| `renderer` | WebGPU рендеринг (SDF-шейпы, WGSL шейдеры, ECS на bitECS) |
| `tanks` | Игровое окружение: физика (Rapier2D), ECS, танки/оружие/снаряды, плагины для RL |
| `ml-common` | Мост между игрой и ML: конвертация состояний в тензоры, конфиг гиперпараметров, curriculum |
| `ml` | Модели, кастомные слои, PPO-тренировка |
| `lib/` | Общие утилиты: математика, шум, RxJS, структуры данных |

---

## Архитектура системы обучения

```
┌──────────────────────────────────────────────────────────┐
│                    Main Thread                           │
│           LearnerManager (оркестратор)                   │
│  - Агрегирует сэмплы от акторов в батчи                 │
│  - Считает V-Trace targets и advantages                 │
│  - Управляет reward_ratio (EMA-адаптивный скейлинг)     │
│  - Раздаёт батчи learner-воркерам                       │
└──────────┬──────────────────────────────┬────────────────┘
           │ samples                      │ batches
           ↑                              ↓
┌──────────────────────┐    ┌──────────────────────────────┐
│  Actor Workers (×4)  │    │    Learner Workers (×2)      │
│                      │    │                              │
│  Запускают эпизоды   │    │  Policy Learner:             │
│  игры параллельно,   │    │    SPO loss + entropy        │
│  собирают переходы:  │    │    KL-мониторинг → LR adjust │
│  (s, a, r, logπ, d)  │    │                              │
│                      │    │  Value Learner:              │
│  Нет обучения —      │    │    Clipped MSE loss          │
│  только сбор данных  │    │    Отдельный LR              │
└──────────────────────┘    └──────────────────────────────┘
```

---

## Нейросетевая архитектура (v10)

**Две отдельные сети**, обучаются параллельно:

### Policy Network (Actor)
- **Dim:** 64, **Heads:** 4, **Depth:** 1
- **Выход:** 4 категориальные головы → `[15, 15, 2, 31]` (стрельба, движение, поворот, поворот башни)

### Value Network (Critic)
- **Dim:** 32, **Heads:** 1, **Depth:** 0.5
- **Выход:** скаляр (оценка состояния)

### Архитектура сети (Perceiver + Transformer + MoE)

```
Входные токены (проекция в dModel через Linear):
├── Tank        [8 фич + type embedding]
├── Turrets     [MAX_TURRETS × 4 фичи]
├── Rays        [RAYS_COUNT × 6 фич]
├── Enemies     [MAX_ENEMIES × 9 фич + type embedding]
├── Allies      [MAX_ALLIES × 9 фич + type embedding]
└── Bullets     [MAX_BULLETS × 4 фичи]

Stage 1 — Vehicle Summarization:
  Perceiver (3×depth): latent Q [4, dim], KV = [allies + enemies]
  → Self-Transformer (3×depth) с MoE FFN

Stage 2 — Ray Summarization:
  Perceiver (4×depth): latent Q [4, dim], KV = rays
  → Self-Transformer (4×depth)

Stage 3 — Projectile Summarization:
  Perceiver (1×depth): latent Q [2, dim], KV = bullets
  → Self-Transformer (1×depth)

Stage 4 — Head Summarization:
  Concat [tank, turret, vehicles_summ, rays_summ, projectiles_summ]
  → Perceiver (4×depth) → Self-Transformer (4×depth)

Выход → [4, dim] → 4 отдельных головы → logits
```

### Кастомные слои

| Слой | Назначение |
|------|-----------|
| **MoELayer** | Mixture of Experts: top-K роутинг (K=2, 16 экспертов), SiLU, jitter noise, expert/router dropout |
| **MultiHeadAttentionLayer** | Scaled dot-product attention с масками |
| **PerceiverLayer** | Cross-attention (latent queries ↔ input KV) + self-attention цепочки |
| **NoisyDenseLayer** | Pink (colored) noise для exploration |
| **RMSNormLayer** | Root Mean Square нормализация |
| **VariableLayer** | Обучаемая константа (learnable latent queries) |
| **topK / gatherND** | Кастомные ops с поддержкой градиентов (для MoE роутинга) |
| **SwinAttentionLayer** | Оконное иерархическое внимание |
| **CrossTransformer** | Кросс-модальное внимание |

---

## Пайплайн обучения (PPO/APPO)

### Фаза 1: Сбор данных (Actor Workers)

- 4 воркера параллельно крутят эпизоды (~6000 фреймов ≈ 3 мин симуляции)
- Каждый фрейм: состояние → Policy Network → действие → награда
- Собирают переходы: `(state, action, reward, logπ_old, logits, done)`

### Фаза 2: V-Trace (off-policy коррекция)

```
Для t от T-1 до 0:
  ρ_t = exp(logπ_current - logπ_behavior)
  δ_t = r_t + γ·V(s_{t+1}) - V(s_t)

  ρ̄_t = min(ρ_t, clipRho)
  c̄_t = min(ρ_t, clipC)

  v̂_t = V(s_t) + ρ̄_t·δ_t + γ·c̄_t·(v̂_{t+1} - V(s_{t+1}))
  A_t = ρ̄_t^PG · δ_t + γ·c̄_t·A_{t+1}
```

### Фаза 3: Обучение Policy Network

```
Для каждого эпоха (policyEpochs):
  Для каждого мини-батча (miniBatchSize):
    logits = PolicyNetwork(states)
    logπ_new = Σᵢ log_softmax(logits[i])[action[i]]
    r = exp(logπ_new - logπ_old)

    // SPO Loss (упрощённый PPO)
    quad = (r - 1)² / (2 · clipRatio)
    loss_policy = -mean(A·r - |A|·quad)

    // Entropy регуляризация
    loss_entropy = -entropyCoeff · mean(Σ p·log(p))

    // Total
    loss = loss_policy + loss_entropy

    Gradient clipping (global norm ≤ 1.0)
    AdamW step

  // Early stopping по KL
  if KL > kl_high → прерываем эпохи
```

### Фаза 4: Обучение Value Network

```
Для каждого эпоха (valueEpochs):
  Для каждого мини-батча:
    v_new = ValueNetwork(states)
    v_clipped = v_old + clip(v_new - v_old, -clipRatio, clipRatio)

    loss = mean(max((returns - v_new)², (returns - v_clipped)²)) · lossCoeff

    Gradient clipping → AdamW step
```

### Фаза 5: Адаптация Learning Rate

- Динамический LR на основе истории KL-дивергенции
- Диапазон: `[1e-5, 1e-3]`, начальный: `1e-4`
- Если KL слишком высокий → снижаем LR
- Если KL слишком низкий → повышаем LR

---

## Оптимизатор

**AdamW** (кастомный, на основе PatchedAdamOptimizer):
- Decoupled weight decay: `1e-6`
- Исключения из weight decay: bias, LayerNorm, RMSNorm, NoisyDense sigma
- Gradient clipping: global norm ≤ 1.0

---

## Система наград

```typescript
Веса:
  KILL_REWARD:           1.2    // Убийство врага
  HIT_REWARD:            0.05   // Попадание
  ADJACENT_ENEMY_REWARD: 0.3    // Нахождение рядом с врагом
  EXPLORATION_WITH_ENEMY:    0.06  // Исследование с врагом поблизости
  EXPLORATION_WITHOUT_ENEMY: 0.02  // Исследование без врага
  PROXIMITY_PENALTY:     0.005  // Штраф за столкновения

Финальная награда:
  if successRatio ≤ 0.1 → 0
  else:
    proportion = clamp(myScore / teamScore, 0, 1)
    reward = successRatio × proportion × FINAL_REWARD_POOL × teamCount
```

Адаптивное масштабирование: `reward_ratio` (EMA) нормализует rewards для стабильности advantages.

---

## Входное пространство

| Группа | Размер | Описание |
|--------|--------|----------|
| Tank | [8] | HP, x, y, vx, vy, turret_cos, turret_sin, radius |
| Turrets | [MAX_TURRETS, 4] | угол, перезарядка, тип, боезапас |
| Rays | [RAYS_COUNT, 6] | local_xy, direction_xy, дистанция, тип попадания |
| Enemies | [MAX_ENEMIES, 9] | как tank + type embedding, с маской |
| Allies | [MAX_ALLIES, 9] | как enemies, с маской |
| Bullets | [MAX_BULLETS, 4] | x, y, vx, vy, с маской |

---

## Пространство действий

4 категориальные головы (multi-discrete):

| Голова | Размер | Действие |
|--------|--------|----------|
| 0 | 15 | Стрельба |
| 1 | 15 | Движение |
| 2 | 2 | Поворот |
| 3 | 31 | Поворот башни |

---

## Гиперпараметры

| Параметр | Значение |
|----------|---------|
| γ (discount) | 0.99 |
| Policy clip ratio | 0.2 |
| Value clip ratio | 0.2 |
| Entropy coeff | 0.01–0.1 (убывает) |
| Batch size | 4096 (256 × 16) |
| Mini-batch | 256 |
| Policy epochs | 4 |
| Value epochs | 4 |
| Gradient clip norm | 1.0 |
| Weight decay | 1e-6 |
| Initial LR | 1e-4 |
| LR range | [1e-5, 1e-3] |
| Episode length | ~6000 фреймов (~3 мин) |
| Actor workers | 4 |
| Learner workers | 2 (policy + value) |

---

## Curriculum Learning

Система динамической генерации сценариев:
- Множество пресетов: 1v1, 3v3, AgentsVsBots, Diagonal, Grid и др.
- Отслеживание success ratio по каждому сценарию
- Адаптация сложности на основе прогресса обучения
- Batch size и gamma могут зависеть от итерации curriculum

---

## MoE (Mixture of Experts) — ветка feat/moe

- **Роутер:** Linear [inputDim → numExperts] → softmax → top-K
- **Эксперты:** up-projection → SiLU → down-projection (16 экспертов, K=2)
- **Регуляризация:** jitter noise, router/expert dropout (без auxiliary loss)
- **Интеграция:** заменяет стандартный FFN в каждом Transformer блоке
- **Градиенты:** кастомные topK и gatherND с поддержкой backprop

---

## Ключевые файлы

| Файл | Назначение |
|------|-----------|
| `ml/src/PPO/index.ts` | Точка входа: инициализация воркеров |
| `ml/src/PPO/train.ts` | V-Trace, policy/value loss, KL |
| `ml/src/PPO/Learner/createLearnerManager.ts` | Оркестрация батчей |
| `ml/src/PPO/Learner/createPolicyLearnerAgent.ts` | Цикл обучения policy |
| `ml/src/PPO/Learner/createValueLearnerAgent.ts` | Цикл обучения value |
| `ml/src/PPO/Actor/EpisodeManager.ts` | Генерация эпизодов |
| `ml/src/Models/Networks/v10.ts` | Архитектура Perceiver+Transformer+MoE |
| `ml/src/Models/Layers/MoELayer.ts` | Mixture of Experts |
| `ml/src/Models/Create.ts` | Фабрика моделей |
| `ml/src/Models/Optimizer/AdamW.ts` | Кастомный AdamW |
| `ml/src/Reward/calculateReward.ts` | Расчёт наград |
| `ml-common/config.ts` | Гиперпараметры |
| `ml-common/InputArrays.ts` | Конвертация game state → тензоры |
| `ml-common/Curriculum/` | Генерация сценариев |
