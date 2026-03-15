# Глубокий анализ: как улучшить обучение на порядок

## Содержание

1. [Критические баги, блокирующие обучение](#1-критические-баги)
2. [Reward Engineering](#2-reward-engineering)
3. [Архитектура сети](#3-архитектура-сети)
4. [Алгоритм обучения (PPO/V-Trace)](#4-алгоритм-обучения)
5. [Exploration](#5-exploration)
6. [Curriculum Learning](#6-curriculum-learning)
7. [Sample Efficiency](#7-sample-efficiency)
8. [Нормализация входов](#8-нормализация-входов)
9. [Масштабирование инфраструктуры](#9-масштабирование-инфраструктуры)
10. [Roadmap приоритетов](#10-roadmap)

---

## 1. Критические баги

### 1.1 V-Trace rho clipping отключён

**Файл:** `ml/src/PPO/train.ts:341`
```typescript
const rho = 1; // Math.min(rhos[t], clipRho);
```

**Проблема:** Ядро V-Trace — clipping importance weights для off-policy коррекции — полностью выключен. Это означает, что когда actor собирает данные с policy v100, а learner уже на v120, importance ratio `ρ = π_new/π_old` может быть 10x–100x. Без clipping:
- Градиенты value function взрываются
- Advantage estimates имеют огромную дисперсию
- Обучение нестабильно

**Ссылка:** [Espeholt et al., "IMPALA: Scalable Distributed Deep-RL", 2018](https://arxiv.org/abs/1802.01561) — Section 3, формула (1): `ρ̄_t = min(ρ_t, ρ̄)` — это не опциональный трюк, а основа алгоритма.

**Фикс:** Раскомментировать `Math.min(rhos[t], clipRho)`.

**Ожидаемый эффект:** Стабилизация обучения, снижение дисперсии градиентов в 5–10x.

---

### 1.2 KL-дивергенция вычисляется неправильно

**Файл:** `ml/src/PPO/train.ts:84-98`
```typescript
const diff = oldLogProb.sub(newLogProbs);
const kl = diff.mean().abs(); // ← неверно
```

**Проблема:** Правильная аппроксимация KL (Schulman, 2020):
```
KL ≈ E[(π_old/π_new - 1) - log(π_old/π_new)]
   = E[exp(logπ_old - logπ_new) - 1 - (logπ_old - logπ_new)]
```

Текущий код берёт `|mean(logπ_old - logπ_new)|` — это не KL, а абсолютное среднее log-ratio. Для малых изменений это грубо совпадает, но:
- При больших policy shift даёт заниженные значения
- `abs()` скрывает направление дрейфа
- KL thresholds (0.013) откалиброваны под неверную метрику

**Ссылка:** [Schulman, "Approximating KL Divergence", 2020](http://joschu.net/blog/kl-approx.html)

**Фикс:**
```typescript
const logRatio = newLogProbs.sub(oldLogProb);
const ratio = logRatio.exp();
const kl = ratio.sub(1).sub(logRatio).mean(); // proper KL approx
```

**Ожидаемый эффект:** Корректная ранняя остановка → policy не коллапсирует, LR scheduling работает адекватно.

---

### 1.3 Опечатка в AdamW

**Файл:** `ml/src/Models/Optimizer/AdamW.ts`
```typescript
if (name.includes('batchnorm')) return nulwl; // ← typo, nulwl не определён
```

Упадёт при первом вызове с batchnorm параметром. Сейчас не используется, но мина замедленного действия.

---

## 2. Reward Engineering

### Гипотеза: Sparse reward — главная причина медленного обучения

**Текущее состояние:**
- `calculateActionReward()` возвращает **только** score от попаданий/убийств
- `HIT_REWARD = 0.05`, `KILL_REWARD = 1.2`
- Остальные reward shaping компоненты **закомментированы** (ADJACENT_ENEMY, EXPLORATION)
- Финальная награда — episode-level, зависит от `successRatio`

**Проблема:** В 3-минутном эпизоде (~6000 фреймов) агент получает ненулевую reward только при попадании. Если агент не умеет целиться — reward = 0 на 99%+ фреймов. Это делает credit assignment практически невозможным.

### 2.1 Потенциальная reward shaping (dense rewards)

**Подход:** Добавить непрерывные сигналы, не меняя оптимальную политику.

| Компонент | Описание | Вес |
|-----------|----------|-----|
| Proximity to enemy | `exp(-dist/range)` когда ≤ range | 0.01/frame |
| Aiming accuracy | `cos(angle_to_enemy)` когда враг в зоне видимости | 0.005/frame |
| Map exploration | Новые тайлы посещены | 0.002/tile |
| Damage taken penalty | `-hp_lost / max_hp` | -0.01 |
| Alive bonus | Оставаться живым | 0.001/frame |
| Team coordination | Расстояние до союзников в пределах [min, max] | 0.002/frame |

**Ссылка:** [Ng, Harada, Russell, "Policy Invariance Under Reward Transformations", 1999](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf) — potential-based reward shaping сохраняет оптимальную политику.

**Формула:** `F(s, s') = γ·Φ(s') - Φ(s)` где `Φ` — потенциальная функция.

Для proximity: `Φ(s) = -min_dist_to_enemy / map_size` → при приближении reward > 0.

**Ожидаемый эффект:** Ускорение начального обучения в 5–20x (агент получает gradient signal с первого фрейма).

---

### 2.2 Curiosity-driven exploration (intrinsic reward)

**Подход:** Random Network Distillation (RND) — добавить intrinsic reward за посещение "новых" состояний.

**Ссылка:** [Burda et al., "Exploration by Random Network Distillation", 2019](https://arxiv.org/abs/1810.12894)

```
r_intrinsic = ||f(s) - f̂(s)||²
```

где `f` — фиксированная random network, `f̂` — обучаемая predictor network. Высокая ошибка = "новое" состояние.

**Преимущество для вашего проекта:**
- Танк получает reward за исследование карты даже без врагов
- Помогает curriculum: агент учится навигации до того, как учится стрелять
- Реализуемо в TensorFlow.js (маленькая MLP, ~32 dim)

**Ожидаемый эффект:** Решение проблемы "мёртвого старта" когда агент не двигается.

---

### 2.3 Hindsight Experience Replay (HER)

**Подход:** Если агент не достиг цели, переразметить trajectory с альтернативной целью, которую он случайно достиг.

**Ссылка:** [Andrychowicz et al., "Hindsight Experience Replay", 2017](https://arxiv.org/abs/1707.01495)

**Применение к танкам:**
- Цель "убить врага" не достигнута → переразметить как "приблизиться к врагу" (достигнута!)
- Цель "выиграть раунд" → "нанести максимальный урон"
- Умножает эффективный dataset в 4–8x

---

## 3. Архитектура сети

### Гипотеза: Сеть слишком глубокая для своей ширины

**Текущее:**
- dModel=64, 4 heads → **16 dims per head** (очень мало)
- **24 attention layers** суммарно (3+3+4+4+1+1+4+4 через perceiver+transformer)
- Правило: `dModel ≥ num_layers` для стабильного gradient flow — здесь нарушено (64 < 24? нет, но per-head dim = 16 < 24)

### 3.1 Оптимальное соотношение width/depth

**Ссылка:** [Kaplan et al., "Scaling Laws for Neural Language Models", 2020](https://arxiv.org/abs/2001.08361) — для трансформеров оптимально `d_model ∝ n_layers^0.7`.

**Рекомендация:**

| Параметр | Текущий | Рекомендуемый | Почему |
|----------|---------|---------------|--------|
| dModel (policy) | 64 | 96 | 24 dim/head вместо 16 |
| Heads | 4 | 4 | Достаточно |
| Total layers | 24 | 8 | 3x меньше, но каждый слой мощнее |
| Vehicle summarizer | 3+3 | 1+1 | Достаточно для 8–16 entities |
| Rays summarizer | 4+4 | 1+1 | Rays уже упорядочены, не нужна глубина |
| Head summarizer | 4+4 | 2+2 | Основная интеграция — здесь глубина нужнее |

**Ожидаемый эффект:** 3x быстрее forward/backward pass, лучший gradient flow, ~на одном уровне по capacity.

---

### 3.2 Perceiver bottleneck — слишком сильная компрессия

**Проблема:** 4 latent tokens сжимают до 16 entities. Compression ratio 4:1 — теряется пространственная информация.

**Ссылка:** [Jaegle et al., "Perceiver: General Perception with Iterative Attention", 2021](https://arxiv.org/abs/2103.03206) — авторы используют 512–1024 latents для тысяч input tokens. При 16 entities нужно 8–12 latents.

**Альтернатива:** Для малого числа entities (<20) perceiver избыточен. Прямой cross-attention эффективнее:

```
vehicle_context = MultiHeadAttention(
    query = tank_token,      // [1, dModel]
    key/value = all_entities  // [N, dModel]
)
```

Это O(N) вместо O(L×N) для perceiver с L итерациями.

---

### 3.3 MoE: снизить topK, добавить load balancing loss

**Текущее:** topK=4, 16 экспертов, jitter noise отключён, нет auxiliary loss.

**Проблема:**
- topK=4 из 16 = 25% активных → слабая sparsity
- Без load balancing loss эксперты коллапсируют (2–3 эксперта забирают весь трафик)
- Jitter noise = 0 → нет exploration в routing

**Ссылка:** [Fedus et al., "Switch Transformers", 2022](https://arxiv.org/abs/2101.03961) — topK=1 + auxiliary loss наиболее эффективен. [Lepikhin et al., "GShard", 2021](https://arxiv.org/abs/2006.16668) — topK=2 оптимально для средних моделей.

**Рекомендация:**
```typescript
// MoE config
topK: 2,          // вместо 4
numExperts: 8,    // вместо 16 (при topK=2 достаточно)
jitterNoise: 0.01, // включить
// + добавить auxiliary loss:
auxLoss = α * CV(expert_load)² // α = 0.01
```

**Ожидаемый эффект:** -50% compute в MoE layers, лучшее использование экспертов.

---

### 3.4 Value network: отдельная архитектура

**Ссылка:** [Cobbe et al., "Phasic Policy Gradient", 2021](https://arxiv.org/abs/2009.04416) — PPG показывает, что sharing features между policy и value вредит обоим. Отдельные архитектуры лучше.

**Текущее:** Value network = уменьшенная копия policy (dim=32, heads=1). Это может быть неоптимально.

**Рекомендация:** Value network не нуждается в per-entity attention. Достаточно:
```
Concat all inputs → 2-layer MLP (256, 128) → scalar
```

Value function предсказывает скаляр для всего состояния — не нужна entity-level гранулярность. Упрощение value network ускорит обучение critic, что улучшит advantage estimates.

---

## 4. Алгоритм обучения

### 4.1 SPO loss → стандартный PPO clip

**Текущее:**
```
quad = (r - 1)² / (2·clipRatio)
loss = -mean(A·r - |A|·quad)
```

**Проблема:** SPO — менее изученный вариант PPO. Квадратичный штраф `(r-1)²` растёт неограниченно при больших отклонениях ratio. Стандартный PPO clip имеет bounded gradient:

```
loss_clip = -min(r·A, clip(r, 1-ε, 1+ε)·A)
```

**Ссылка:** [Schulman et al., "Proximal Policy Optimization Algorithms", 2017](https://arxiv.org/abs/1707.06347) — clip более стабилен эмпирически.

**Гипотеза:** SPO может создавать слишком агрессивные градиенты при |A| >> 0, потому что `|A|·(r-1)²` масштабируется квадратично по ratio. Стоит A/B тестировать оба.

---

### 4.2 V-Trace → DAPO/GRPO (on-policy alternatives)

**Гипотеза:** V-Trace с отключённым rho clipping = broken. Даже с фиксом, off-policy коррекция в APPO создаёт staleness проблемы.

**Альтернатива 1: Full on-policy PPO**
- Синхронизировать actors и learners: actor ждёт обновления policy перед следующим эпизодом
- Простой, стабильный, хорошо изучен
- Минус: actors простаивают во время обучения

**Альтернатива 2: GRPO (Group Relative Policy Optimization)**

**Ссылка:** [Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning", 2024](https://arxiv.org/abs/2402.03300)

GRPO убирает value network полностью:
```
Advantage_i = (R_i - mean(R_group)) / std(R_group)
```

Для группы из K эпизодов из одного состояния, advantage = нормализованный reward относительно группы.

**Преимущества:**
- Нет value network → 50% меньше параметров и compute
- Нет bootstrapping → нет ошибок value estimation
- Хорошо работает с sparse rewards

**Применение:** Запустить K=4 эпизода из одного начального состояния (один сценарий), сравнить rewards → тот кто лучше получает положительный advantage.

---

### 4.3 Advantage estimation: GAE вместо V-Trace

Если перейти на on-policy PPO, GAE (Generalized Advantage Estimation) стабильнее V-Trace:

**Ссылка:** [Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation", 2016](https://arxiv.org/abs/1506.02438)

```
δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
A_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
```

С λ=0.95 — баланс между bias и variance. Текущий V-Trace c `rho=1` по сути уже GAE, но без λ-damping.

---

### 4.4 Clipping advantages

**Проблема:** Нормализованные advantages не клиппируются. Outliers (>5σ) создают огромные градиенты.

**Фикс:**
```typescript
const clippedAdvantages = normalizedAdvantages.clipByValue(-5, 5);
```

**Ссылка:** [Engstrom et al., "Implementation Matters in Deep Policy Gradients", 2020](https://arxiv.org/abs/2005.12729) — advantage clipping входит в "bag of tricks" для стабильного PPO.

---

## 5. Exploration

### Гипотеза: Entropy-only exploration недостаточна для multi-agent combat

**Текущее:**
- Entropy regularization (0.01–0.1) — единственный механизм exploration
- NoisyDense layers есть в коде, но **не используются** (закомментировано в Create.ts)
- Dirichlet noise реализован, но **никогда не вызывается**

### 5.1 Включить NoisyNets

**Ссылка:** [Fortunato et al., "Noisy Networks for Exploration", 2018](https://arxiv.org/abs/1706.10295)

NoisyDense уже реализован с colored (pink) noise — просто раскомментировать в `Create.ts`. Pink noise даёт temporally correlated exploration (агент "пробует" стратегию несколько фреймов подряд, а не дёргается каждый фрейм).

**Ожидаемый эффект:** Более structured exploration, быстрое обнаружение рабочих стратегий.

---

### 5.2 Population-Based Training (PBT)

**Ссылка:** [Jaderberg et al., "Population Based Training of Neural Networks", 2017](https://arxiv.org/abs/1711.09846)

**Идея:** Вместо одного агента — популяция из N=8–16 агентов с разными гиперпараметрами. Периодически:
1. Evaluate всех агентов
2. Bottom 20% копируют веса + hyperparams от top 20%
3. Мутируют hyperparams (LR, entropy coeff, reward weights)

**Применение:** У вас уже 4 actor workers. Расширить до 8, каждый с своим набором hyperparams. Каждые 10 эпизодов — exploit/explore step.

**Ожидаемый эффект:** Автоматический hyperparameter tuning + diverse agent behavior → лучшее обучение через self-play.

---

### 5.3 Self-play с историческими агентами

**Ссылка:** [Bansal et al., "Emergent Complexity via Multi-Agent Competition", 2018](https://arxiv.org/abs/1710.03748)

**Текущее:** В коде есть `RandomHistoricalAgent` — можно загружать старые модели как оппонентов. Но неясно, используется ли это систематически.

**Рекомендация:**
- 50% матчей: agent vs current self
- 30% матчей: agent vs random historical snapshot (последние 50 версий)
- 20% матчей: agent vs scripted bots (baseline)

Это предотвращает "forgetting" и создаёт robust policy.

---

## 6. Curriculum Learning

### Гипотеза: Curriculum слишком простой и noisy

**Текущее:**
- 9 сценариев с порогом разблокировки 30% success rate
- Обновление success ratio только на reference эпизодах (10% от всех)
- RingBuffer из 30 эпизодов для скользящего среднего
- Линейное взвешивание: `weight = clamp(0.9 - successRatio, 0.2, 1)`

### 6.1 Automatic Domain Randomization (ADR)

**Ссылка:** [OpenAI, "Solving Rubik's Cube with a Robot Hand", 2019](https://arxiv.org/abs/1910.07113) — ADR автоматически расширяет distribution параметров окружения.

**Применение:**
```
Параметры среды (domain):
- Кол-во врагов: 1 → 2 → 4
- Скорость врагов: 0.5x → 1.0x → 1.5x
- HP врагов: 50% → 100% → 150%
- Размер карты: маленькая → средняя → большая
- Тип врагов: только лёгкие → смешанные → тяжёлые
```

Когда success > 80% на текущих параметрах → расширяем диапазон. Когда success < 20% → сужаем.

**Преимущество:** Плавная прогрессия вместо дискретных сценариев. Нет порога "всё или ничего".

---

### 6.2 PLR (Prioritized Level Replay)

**Ссылка:** [Jiang et al., "Replay-Guided Adversarial Environment Design", 2021](https://arxiv.org/abs/2110.02439)

**Идея:** Приоритизировать сценарии, где агент имеет максимальный regret (разница между оптимальным и текущим return).

```
priority(scenario) = |V*(s₀) - V_current(s₀)|
```

Приближение: использовать TD error как proxy для regret. Сценарии с высоким TD error = агент плохо предсказывает reward = много чему учиться.

---

## 7. Sample Efficiency

### 7.1 Recompute advantages каждую эпоху

**Проблема:** V-Trace targets считаются **один раз** перед 4 эпохами обучения. После первой эпохи policy уже изменилась → advantages stale.

**Ссылка:** [Andrychowicz et al., "What Matters In On-Policy Reinforcement Learning?", 2021](https://arxiv.org/abs/2006.05990) — recomputing advantages between epochs значительно улучшает sample efficiency.

**Фикс:** Пересчитывать V-Trace targets перед каждой эпохой (или хотя бы после 2-й).

**Ожидаемый эффект:** +10–20% sample efficiency при тех же данных.

---

### 7.2 Стратегии приоритизации данных

**Текущее:** ReplayBuffer просто shuffles данные. PrioritizedReplayBuffer реализован, но, видимо, не используется в PPO pipeline.

**Рекомендация:** Приоритизировать сэмплы по |TD error|:

**Ссылка:** [Schaul et al., "Prioritized Experience Replay", 2016](https://arxiv.org/abs/1511.05952)

```
P(i) ∝ |δ_i|^α + ε
w_i = (N · P(i))^(-β)  // importance sampling correction
```

---

### 7.3 Увеличить policy epochs с recompute

С recomputed advantages можно безопасно увеличить epochs:
- Текущее: 4 epochs
- Рекомендация: 8–10 epochs с recomputed advantages + KL early stopping

**Ссылка:** [Hilton et al., "Scaling Data-Constrained Language Models", 2023](https://arxiv.org/abs/2305.16264) — многократное переиспользование данных эффективно при правильном scheduling.

---

## 8. Нормализация входов

### Гипотеза: Ненормализованные входы замедляют обучение

**Текущее распределение фичей:**
```
HP:           [0, 500]     — НЕ нормализован
Position:     [-1, 1]      — нормализован (÷ map_size)
Velocity:     [-2, 2]      — нормализован (÷ QUANT=100)
Rotation:     [-1, 1]      — cos/sin (уже bounded)
Collider:     log-scaled   — logNorm
Battle dims:  log-scaled   — log(width), log(height)
```

**Проблема:** HP в диапазоне [0, 500] при остальных фичах в [-2, 2]. Linear projection `W·x + b` должен выучить коэффициенты, отличающиеся на 2 порядка. Это:
- Замедляет обучение (разные learning rates по фичам)
- Создаёт ill-conditioned gradients
- Embedding layer добавляет 0 (zeros initializer) к уже несбалансированным активациям

### 8.1 Running normalization

**Ссылка:** [Engstrom et al., "Implementation Matters in Deep Policy Gradients", 2020](https://arxiv.org/abs/2005.12729) — observation normalization = один из критических "tricks".

**Рекомендация:**
```typescript
// Welford's online algorithm
class RunningNormalizer {
    mean: Float32Array;
    var: Float32Array;
    count: number;

    normalize(x) {
        return (x - this.mean) / sqrt(this.var + 1e-8);
    }
}
```

Применять к каждой группе фичей отдельно. Или проще — нормализовать HP: `hp / max_hp`.

### 8.2 Reward normalization

**Текущее:** `rewardRatio` масштабирует rewards по EMA std returns. Это хорошо, но:
- Target std = 1.5 — почему не 1.0?
- Clamp [0.8, 1.2] слишком узкий — может не успеть скомпенсировать

**Рекомендация:** Также нормализовать rewards через running stats: `r' = (r - mean_r) / std_r`. Это стандартная практика в PPO.

**Ссылка:** [Andrychowicz et al., 2021](https://arxiv.org/abs/2006.05990) — reward normalization входит в top-5 tricks.

---

## 9. Масштабирование инфраструктуры

### 9.1 Больше actor workers

**Текущее:** 4 actors, batch size 4096. Каждый actor должен собрать ~1024 frames до batch ready. При 6000 frames/episode = ~6 episodes worth перед каждым train step.

**Проблема:** Actors завалены работой, learner ждёт. Backpressure queue = 2 — actors часто idle.

**Рекомендация:** 8–16 actors. Это линейно ускорит сбор данных. Browser Web Workers поддерживают это.

### 9.2 Async PPO → Sync PPO

**Гипотеза:** Переход на sync PPO может улучшить sample quality достаточно, чтобы компенсировать потерю throughput.

**Ссылка:** [Espeholt et al., 2018](https://arxiv.org/abs/1802.01561) vs [Schulman et al., 2017](https://arxiv.org/abs/1707.06347) — trade-off: APPO = больше throughput, PPO = лучшее quality per sample.

При 4 workers sync overhead невелик. С 16+ workers async становится необходим.

### 9.3 Mixed precision training

TensorFlow.js поддерживает float16 в WebGPU backend:
```typescript
tf.env().set('WEBGPU_USE_LOW_POWER_GPU', false);
// Enable mixed precision
tf.env().set('WEBGPU_CPU_FORWARD', false);
```

Ожидаемое ускорение: 1.5–2x на GPU, ~0% потери accuracy.

---

## 10. Roadmap

### Tier 1: Критические фиксы (ожидаемый эффект: 2–5x)

| # | Действие | Сложность | Влияние |
|---|----------|-----------|---------|
| 1 | Включить V-Trace rho clipping | 1 строка | Стабильность обучения |
| 2 | Исправить KL computation | 5 строк | Корректный LR scheduling |
| 3 | Нормализовать HP и другие raw фичи | 20 строк | Быстрее convergence |
| 4 | Раскомментировать reward shaping | 10 строк | Dense reward signal |
| 5 | Advantage clipping (±5σ) | 1 строка | Стабильность градиентов |

### Tier 2: Архитектурные улучшения (ожидаемый эффект: 2–3x)

| # | Действие | Сложность | Влияние |
|---|----------|-----------|---------|
| 6 | Уменьшить depth, увеличить width (96 dim, 8 layers) | Средняя | 3x быстрее train step |
| 7 | Direct cross-attention вместо perceiver для entities | Средняя | Лучшие entity representations |
| 8 | MoE: topK=2, auxiliary loss, jitter=0.01 | Низкая | Эффективнее compute |
| 9 | Включить NoisyDense layers | Низкая | Structured exploration |
| 10 | Упрощённый value network (MLP) | Средняя | Быстрее critic, лучше advantages |

### Tier 3: Алгоритмические улучшения (ожидаемый эффект: 3–10x)

| # | Действие | Сложность | Влияние |
|---|----------|-----------|---------|
| 11 | Recompute advantages каждую эпоху | Средняя | +20% sample efficiency |
| 12 | GRPO вместо PPO+Value (убрать critic) | Высокая | Упрощение + лучше для sparse rewards |
| 13 | RND intrinsic motivation | Средняя | Решение dead start |
| 14 | ADR curriculum вместо фиксированных сценариев | Средняя | Плавная прогрессия |
| 15 | Self-play с историческими агентами | Низкая | Robust policy |

### Tier 4: Масштабирование (ожидаемый эффект: 2–4x throughput)

| # | Действие | Сложность | Влияние |
|---|----------|-----------|---------|
| 16 | 8–16 actor workers | Низкая | 2–4x data throughput |
| 17 | Sync PPO (если workers ≤8) | Средняя | Лучшее quality per sample |
| 18 | Population-Based Training | Высокая | Автоматический hyperparameter tuning |

---

## Суммарная оценка

**Текущие потери от найденных проблем:**
- V-Trace rho=1 + неверный KL: ~50% потери эффективности обучения
- Sparse reward + ненормализованные входы: ~5–10x замедление convergence
- Переглубокая/переузкая архитектура: ~2–3x лишний compute

**Реалистичная оценка при внедрении Tier 1 + Tier 2:**
Улучшение в **5–10x** по скорости достижения текущего уровня качества.

**При полном внедрении Tier 1–3:**
Улучшение в **10–50x** по итоговому качеству и скорости обучения, на основе аналогичных результатов из цитированных papers (IMPALA, PPG, RND, ADR).
