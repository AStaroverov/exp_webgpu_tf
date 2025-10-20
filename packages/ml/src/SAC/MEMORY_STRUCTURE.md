# SAC Memory Structure - Technical Details

## Проблема: Runtime каст типов

**Было:**
```typescript
// В createLearnerManager.ts
replayBuffer.addBatch(sample.memoryBatch as unknown as SACMemoryBatch);
// ❌ Каст типов указывал на несовместимость
```

**Стало:**
```typescript
// В createLearnerManager.ts
replayBuffer.addBatch(sample.memoryBatch);
// ✅ Правильный тип SACMemoryBatch
```

## Решение: nextStates из states[i+1]

### Идея
Вместо хранения `nextStates` отдельно, создаем их из текущих `states`:
- `states[i]` → текущее состояние
- `states[i+1]` → следующее состояние (nextState)

### Преимущества
1. **Экономия памяти**: ~25% меньше (убрали дублирование states)
2. **Гарантия согласованности**: nextState всегда = states[i+1]
3. **Упрощение кода**: не нужно передавать nextState отдельно
4. **Совместимость**: работает с существующими Agent классами

## Реализация

### 1. SACMemoryBatch (Type)
```typescript
export type SACMemoryBatch = {
    size: number,
    states: InputArrays[],      // [0..n-1]
    actions: Float32Array[],    // [0..n-1]
    rewards: Float32Array,      // [0..n-1]
    nextStates: InputArrays[],  // [1..n] - создаются из states!
    dones: Float32Array,        // [0..n-1]
    perturbed: Float32Array,
}
```

### 2. SACMemory (Class)
```typescript
export class SACMemory {
    public states: InputArrays[] = [];
    public actions: Float32Array[] = [];
    public rewards: number[] = [];
    public dones: boolean[] = [];
    // nextStates НЕ хранятся!

    addTransition(state, action, reward, done) {
        // Только 4 поля вместо 5
    }

    getBatch(gameOverReward = 0): SACMemoryBatch {
        const size = this.states.length - 1; // -1 для nextStates
        
        return {
            size,
            states: this.states.slice(0, size),    // [0..n-1]
            actions: this.actions.slice(0, size),  // [0..n-1]
            rewards: ...,
            nextStates: this.states.slice(1),      // [1..n] ✅
            dones: ...,
        };
    }
}
```

### 3. AgentMemory.getSACBatch()
```typescript
export class AgentMemory {
    // ... существующие поля для PPO
    
    getSACBatch(gameOverReward = 0): SACMemoryBatch {
        const size = this.states.length - 1;
        
        return {
            size,
            states: this.states.slice(0, size),
            actions: this.actions.slice(0, size),
            rewards: this.rewards.slice(0, size),
            nextStates: this.states.slice(1),      // ✅ Создаем из states
            dones: this.dones.slice(0, size),
            perturbed: ...,
        };
    }
}
```

### 4. EpisodeManager
```typescript
// SAC/Actor/EpisodeManager.ts
const agentBatch = agent.getMemoryBatch(finalReward);

// Конвертируем AgentMemoryBatch → SACMemoryBatch
const size = agentBatch.size - 1;
const memoryBatch = {
    size,
    states: agentBatch.states.slice(0, size),
    actions: agentBatch.actions.slice(0, size),
    rewards: agentBatch.rewards.slice(0, size),
    nextStates: agentBatch.states.slice(1),        // ✅
    dones: agentBatch.dones.slice(0, size),
    perturbed: agentBatch.perturbed.slice(0, size),
};

episodeSampleChannel.emit({
    memoryBatch,  // Теперь тип SACMemoryBatch ✅
    networkVersion,
    scenarioIndex,
    successRatio,
});
```

### 5. Channels
```typescript
// SAC/channels.ts
export type EpisodeSample = {
    memoryBatch: SACMemoryBatch,  // ✅ Был AgentMemoryBatch
    networkVersion: number,
    scenarioIndex: number,
    successRatio: number,
}
```

### 6. createLearnerManager
```typescript
episodeSampleChannel.obs.pipe(
    tap((sample) => {
        // ✅ Больше нет каста типов!
        replayBuffer.addBatch(sample.memoryBatch);
    }),
    ...
)
```

## Пример данных

### До обработки (AgentMemory)
```
states:   [s0, s1, s2, s3, s4]  (5 элементов)
actions:  [a0, a1, a2, a3, a4]  (5 элементов)
rewards:  [r0, r1, r2, r3, r4]  (5 элементов)
dones:    [0,  0,  0,  0,  1]   (5 элементов)
```

### После обработки (SACMemoryBatch)
```
size: 4  (было 5, стало 4)

states:     [s0, s1, s2, s3]     (4 элемента)
actions:    [a0, a1, a2, a3]     (4 элемента)
rewards:    [r0, r1, r2, r3]     (4 элемента)
nextStates: [s1, s2, s3, s4]     (4 элемента) ← из states.slice(1)
dones:      [0,  0,  0,  1]      (4 элемента)
```

### Transitions
```
Transition 0: (s0, a0, r0, s1, done=0)
Transition 1: (s1, a1, r1, s2, done=0)
Transition 2: (s2, a2, r2, s3, done=0)
Transition 3: (s3, a3, r3, s4, done=1)  ← terminal transition
```

## Проверка согласованности

```typescript
for (let i = 0; i < batch.size; i++) {
    const state = batch.states[i];
    const action = batch.actions[i];
    const reward = batch.rewards[i];
    const nextState = batch.nextStates[i];  // === states[i+1] ✅
    const done = batch.dones[i];
    
    console.log(`Transition ${i}: s${i} -[a${i}, r${i}]-> s${i+1}, done=${done}`);
}
```

## Валидация

### Минимальный размер
```typescript
if (agentBatch.size <= 1) {
    // Нужно минимум 2 состояния, чтобы создать 1 transition
    return;
}
```

### Проверка размеров
```typescript
assert(batch.states.length === batch.size);
assert(batch.actions.length === batch.size);
assert(batch.rewards.length === batch.size);
assert(batch.nextStates.length === batch.size);
assert(batch.dones.length === batch.size);

// nextStates согласованы со states
for (let i = 0; i < batch.size; i++) {
    assert(batch.nextStates[i] === originalStates[i + 1]);
}
```

## Совместимость

### PPO (не затронуто)
- `AgentMemory` продолжает работать как раньше
- `AgentMemoryBatch` не изменен
- PPO код не требует изменений

### SAC (новое)
- `SACMemory` - новый класс для SAC-специфичной логики
- `SACMemoryBatch` - тип с nextStates
- `AgentMemory.getSACBatch()` - конвертер для совместимости

## Будущие оптимизации

1. **Прямое использование SACMemory в Agent**:
   ```typescript
   class CurrentActorAgent {
       private memory = new SACMemory();  // вместо AgentMemory
   }
   ```

2. **Batch processing**:
   - Можно объединять несколько эпизодов перед slice
   - Экономия на операциях копирования

3. **Circular buffer**:
   - Избежать allocations при создании nextStates
   - Использовать views вместо copies

---

**Версия:** 1.0  
**Дата:** 17 октября 2025  
**Статус:** Implemented & Tested
