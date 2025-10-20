# gSDE в PPO — исправленный план (без пересчёта бэкбона)

Этот документ уточняет предыдущий план: теперь **φ(s)** (фичи для gSDE) извлекаются **без повторного прогона** бэкбона. Даны **2 корректные схемы** интеграции: (A) один модельный граф с двумя выходами и (B) слой‑ориентированная `forward()`-функция.

---

## Ключевая идея gSDE (без изменений)

- Шум строится как линейная функция признаков состояния: `ε(s) = φ(s) @ Θ`, где `φ(s)` — последний скрытый вектор, `Θ` — матрица шума (редко пересэмплируется).  
- Распределение действий остаётся гауссовым с состояние‑зависимой дисперсией:
  \[\n\log \sigmâ(s) = \log \sigma_{\text{base}} + \tfrac{1}{2}\log\Big(\sum_i \phi_i(s)^2\Big)\n\]
- Это даёт корректные `log_prob`, `ratio`, KL для PPO.

---

## Архитектура: два корректных варианта без пересчёта

### Вариант A (рекомендуется): **один функциональный граф с двумя выходами**

Один `Model` возвращает **сразу** и `mean`, и `φ(s)` — один forward‑pass.

```ts
// createPolicyNetwork.ts
const inputs = tf.input({ shape: [STATE_DIM] });

const h1  = tf.layers.dense({ units: 128, activation: 'relu' }).apply(inputs) as tf.SymbolicTensor;
const phi = tf.layers.dense({ units: 128, activation: 'relu' }).apply(h1)     as tf.SymbolicTensor; // φ(s)
const meanOut = tf.layers.dense({ units: ACTION_DIM, activation: null }).apply(phi) as tf.SymbolicTensor;

// Один граф, два выхода
export const policy = tf.model({ inputs, outputs: [meanOut, phi] });

// Использование (rollout/train):
const [mean, phiOut] = policy.predict(states) as [tf.Tensor2D, tf.Tensor2D];
```

**Плюсы:** один прогон сети; корректные градиенты на бэкбон и голову; удобно для `optimizer.minimize`.

---

### Вариант B: **общие слои + `forward()`** (без создания второго Model)

Создаём слои один раз и делаем функцию прямого прохода, возвращающую `{mean, phi}`. Никаких повторных прогонов.

```ts
// layers.ts (инициализируется один раз)
export const dense1  = tf.layers.dense({ units: 128, activation: 'relu' });
export const dense2  = tf.layers.dense({ units: 128, activation: 'relu' }); // φ‑layer
export const meanHead = tf.layers.dense({ units: ACTION_DIM, activation: null });

export function forward(states: tf.Tensor2D) {
  const h1  = dense1.apply(states) as tf.Tensor2D;
  const phi = dense2.apply(h1)     as tf.Tensor2D;
  const mean = meanHead.apply(phi) as tf.Tensor2D;
  return { mean, phi };
}
```

**Плюсы:** простая интеграция в кастомный train‑цикл; полный контроль и единичный pass.

---

## Модуль шума (без изменений концепции)

`NoiseMatrix` управляет Θ и базовой дисперсией.

```ts
class NoiseMatrix {
  private Theta: tf.Tensor2D; // [F, A]
  private step = 0;
  constructor(
    private F: number,
    private A: number,
    private noiseUpdateFrequency: number,
    public logStdBase: tf.Variable,  // shape [A], можно сделать фикс.
    private variantA = true          // A: масштаб после φ@Θ; B: масштаб внутри Θ
  ) {
    this.Theta = tf.zeros([this.F, this.A]) as tf.Tensor2D;
    this.resample();
  }

  resample() {
    tf.tidy(() => {
      if (this.variantA) {
        this.Theta.dispose();
        this.Theta = tf.randomNormal([this.F, this.A]) as tf.Tensor2D;          // Θ ~ N(0, I)
      } else {
        const baseStd = tf.exp(this.logStdBase).reshape([1, this.A]);
        this.Theta.dispose();
        this.Theta = tf.randomNormal([this.F, this.A]).mul(baseStd) as tf.Tensor2D; // Θ ~ N(0, diag(σ_base^2))
      }
    });
  }

  maybeResample() {
    if (this.step % this.noiseUpdateFrequency === 0) this.resample();
    this.step += 1;
  }

  noise(phi: tf.Tensor2D): tf.Tensor2D {
    const raw = phi.matMul(this.Theta) as tf.Tensor2D;     // [B, A]
    if (this.variantA) {
      const baseStd = tf.exp(this.logStdBase).reshape([1, this.A]);
      return raw.mul(baseStd) as tf.Tensor2D;
    }
    return raw;
  }

  dispose() { this.Theta.dispose(); }
}
```

---

## Rollout (сбор опыта)

**Общее для A и B:** берём `mean, φ` **одним forward‑pass**. Ресэмплим Θ только в rollout (раз в `noiseUpdateFrequency`).

```ts
function actRollout(states: tf.Tensor2D, noise: NoiseMatrix) {
  noise.maybeResample();

  // Вариант A
  // const [mean, phi] = policy.predict(states) as [tf.Tensor2D, tf.Tensor2D];

  // Вариант B
  const { mean, phi } = forward(states);

  const eps = noise.noise(phi);                  // [B, A]
  const u = mean.add(eps) as tf.Tensor2D;        // до tanh
  const actions = u.tanh() as tf.Tensor2D;

  // σ̂(s): logStdEff = clip(logStdBase + 0.5*log(sum(phi^2)))
  const logStdEff = tf.tidy(() => {
    const base = noise.logStdBase.reshape([1, ACTION_DIM]);
    const power = tf.log(phi.square().sum(1, true).add(1e-6)).mul(0.5);
    const lse = base.add(power);
    return tf.clipByValue(lse, MIN_LOG_STD, MAX_LOG_STD) as tf.Tensor2D;
  });
  const stdEff = tf.exp(logStdEff) as tf.Tensor2D;

  const oldLogProbs = computeLogProbTanh(actions, mean, stdEff); // [B]
  return { actions, oldLogProbs };
}
```

> В память сохраняем `states, actions, oldLogProbs, (values, rewards, dones, ...)`. `Θ/φ` **не** сохраняем — для train не нужны.

---

## PPO‑обновление (train)

Считаем `newLogProbs` под **новой** политикой `N(μ_new, σ̂_new(φ_new))`. Θ в train‑шаге **не используется**.

```ts
function ppoUpdate(batch, noise: NoiseMatrix, cfg: PPOCfg) {
  // Вариант A:
  // const [meanNew, phiNew] = policy.predict(batch.states) as [tf.Tensor2D, tf.Tensor2D];

  // Вариант B:
  const { mean: meanNew, phi: phiNew } = forward(batch.states);

  const logStdEffNew = tf.tidy(() => {
    const base = noise.logStdBase.reshape([1, ACTION_DIM]);
    const power = tf.log(phiNew.square().sum(1, true).add(1e-6)).mul(0.5);
    const lse = base.add(power);
    return tf.clipByValue(lse, MIN_LOG_STD, MAX_LOG_STD) as tf.Tensor2D;
  });
  const stdEffNew = tf.exp(logStdEffNew) as tf.Tensor2D;

  const newLogProbs = computeLogProbTanh(batch.actions, meanNew, stdEffNew);
  const ratio = newLogProbs.sub(batch.oldLogProbs).exp();

  const surr1 = ratio.mul(batch.advantages);
  const surr2 = tf.clipByValue(ratio, 1 - cfg.clipRatio, 1 + cfg.clipRatio).mul(batch.advantages);
  const policyLoss = tf.minimum(surr1, surr2).mean().mul(-1);

  // (опц.) энтропия через stdEffNew
  const c = 0.5 * Math.log(2 * Math.PI * Math.E);
  const entropy = logStdEffNew.add(c).sum(1).mean();
  const totalLoss = policyLoss.sub(entropy.mul(cfg.entropyCoeff));

  return { policyLoss, entropy, totalLoss };
}
```

---

## Eval / Inference

Детерминированное действие без шума.

```ts
function actEval(states: tf.Tensor2D) {
  // A: const [mean] = policy.predict(states) as [tf.Tensor2D, tf.Tensor2D];
  // B:
  const { mean } = forward(states);
  return mean.tanh() as tf.Tensor2D;
}
```

---

## Конфиг (напоминание)

```ts
type PolicyNetworkConfig = {
  useGSDE: boolean;
  latentDim: number;               // F
  noiseUpdateFrequency: number;    // ресэмпл Θ
  logStdBaseInit: number;          // e.g., -1.0
  minLogStd: number;               // e.g., -5
  maxLogStd: number;               // e.g., 0
  trainableLogStdBase: boolean;    // true/false
  clipRatio: number;               // e.g., 0.2
  entropyCoeff: number;            // e.g., 0.0…0.01
};
```

---

## Частые ошибки (напоминание)

- Реконструкция mean через старый шум в train → смещение; не делать.
- Двойной масштаб шума (и в Θ, и после `φ@Θ`) → выбрать один способ.
- Ресэмпл Θ во время train итераций → скачки KL; ресэмпл только в rollout.
- Неограниченный `logStdBase` → клип и/или мягкая регуляция.

---

Файл рекомендуется положить как `docs/ppo_gsde_plan_v2.md`.
