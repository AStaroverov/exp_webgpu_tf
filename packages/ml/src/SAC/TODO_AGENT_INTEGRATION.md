# SAC Implementation - Agent Integration Status

## ✅ ВРЕМЕННОЕ РЕШЕНИЕ РЕАЛИЗОВАНО

EpisodeManager теперь конвертирует AgentMemoryBatch в SACMemoryBatch автоматически:
- Использует `agent.getMemoryBatch()` (существующий метод)
- Создает `nextStates` из `states[i+1]`
- Обрезает массивы на 1 элемент для согласованности

## Как это работает сейчас

### EpisodeManager (SAC/Actor/EpisodeManager.ts)
```typescript
const agentBatch = agent.getMemoryBatch(finalReward);

// Конвертируем в SAC формат
const size = agentBatch.size - 1;
const memoryBatch = {
    size,
    states: agentBatch.states.slice(0, size),      // [0..n-1]
    actions: agentBatch.actions.slice(0, size),    // [0..n-1]
    rewards: agentBatch.rewards.slice(0, size),    // [0..n-1]
    nextStates: agentBatch.states.slice(1),        // [1..n] - из текущих states!
    dones: agentBatch.dones.slice(0, size),        // [0..n-1]
    perturbed: agentBatch.perturbed.slice(0, size),
};
```

### Преимущества такого подхода
- ✅ Не требует изменений в Agent классах
- ✅ Экономит память (не хранит nextStates отдельно)
- ✅ Гарантирует согласованность (nextState = следующий state)
- ✅ Работает с существующим кодом PPO

## AgentMemory.getSACBatch()

Также добавлен метод `getSACBatch()` в `AgentMemory` для прямой конвертации:

```typescript
const sacBatch = agent.memory.getSACBatch(finalReward);
// Автоматически создает nextStates из states[i+1]
```

## SACMemory класс

Обновлен для хранения только `(s, a, r, done)`:
- `nextStates` удалены из хранения
- Создаются в `getBatch()` из `states.slice(1)`
- Экономия памяти ~25%

## Agent Integration (Опциональное улучшение в будущем)

### 1. Update Agent Interface
Location: `packages/tanks/src/Pilots/Agents/`

Add method to agents:
```typescript
getSACMemoryBatch(gameOverReward: number): SACMemoryBatch | undefined
```

### 2. Update CurrentActorAgent
Location: `packages/tanks/src/Pilots/Agents/CurrentActorAgent.ts`

Changes needed:
- Use `SACMemory` instead of `AgentMemory`
- Store transitions as `(s, a, r, s', done)` instead of `(s, a, value, logProb)`
- Remove value network calls
- Remove GAE/V-trace computation
- Use `act()` from SAC train.ts instead of PPO

Example:
```typescript
import { SACMemory } from '../../../ml-common/Memory.ts';
import { act } from '../../../ml/src/SAC/train.ts';

class CurrentActorAgent {
    private memory = new SACMemory();
    private previousState?: InputArrays;
    
    act(state: InputArrays) {
        // Sample action using SAC
        const { actions } = act(
            this.policyNetwork,
            state,
            minLogStd,
            maxLogStd,
            false // stochastic for training
        );
        
        // Store previous state for next transition
        this.previousState = state;
        
        return actions;
    }
    
    observe(reward: number, nextState: InputArrays, done: boolean) {
        if (this.previousState) {
            // Add complete transition
            this.memory.addTransition(
                this.previousState,
                this.lastAction,
                reward,
                nextState,
                done
            );
        }
    }
    
    getSACMemoryBatch(gameOverReward: number) {
        return this.memory.getBatch(gameOverReward);
    }
}
```

### 3. Update HistoricalAgent (if used)
Similar changes as CurrentActorAgent

### 4. Update Agent Types
Location: `packages/ml-common/Curriculum/types.ts`

Add to TankAgent interface:
```typescript
getSACMemoryBatch?: (gameOverReward: number) => SACMemoryBatch | undefined;
```

## Notes
- Keep PPO agents for backward compatibility
- Can run PPO and SAC in parallel for A/B testing
- SAC agents should have different naming convention (e.g., `SACActorAgent`)
