# ML Frontend (Actor / Pet Project)

Клиентская часть для распределённого обучения: запускает симуляцию игры, собирает опыт, отправляет на backend, загружает обновлённые модели.

## Цель
Создать браузерное приложение, которое:
1. **Загружает модели** из Supabase Storage
2. **Запускает симуляцию** игры с текущей моделью
3. **Собирает опыт** (states, actions, rewards, dones)
4. **Отправляет батчи** через Ably в канал `ml:experience.v1`
5. **Подписывается на обновления** моделей из Supabase

## План работы

### 1. Ознакомление с текущим решением (packages/ml)
**Цель:** Понять существующую архитектуру клиентской части

**Что изучить:**
- `src/PPO/Actor/` - генерация опыта через симуляцию
- `src/PPO/ActorWorker.ts` - worker для актора
- `src/PPO/VisTest/` - визуализация и отладка
- `src/Reward/` - система наград
- `src/Models/` - работа с моделями (загрузка, применение)
- `appo.ts` / `appo.html` - точка входа приложения

**Ключевые компоненты:**
- `EpisodeManager` - управление эпизодами симуляции
- `VisTestEpisodeManager` - эпизоды с визуализацией
- `channels.ts` - коммуникация через BroadcastChannel
- `ActorWorker` - изолированная генерация опыта

### 2. Перенос клиентской части в ml-frontend
**Цель:** Мигрировать весь код, связанный с генерацией опыта

**Что переносим:**

#### Core Actor Components
```
packages/ml/src/PPO/Actor/
  └─ EpisodeManager.ts          → packages/ml-frontend/src/Actor/EpisodeManager.ts

packages/ml/src/PPO/VisTest/
  └─ VisTestEpisodeManager.ts   → packages/ml-frontend/src/Actor/VisTestEpisodeManager.ts

packages/ml/src/PPO/
  └─ ActorWorker.ts             → packages/ml-frontend/src/Actor/ActorWorker.ts
```

#### Reward System
```
packages/ml/src/Reward/
  └─ calculateReward.ts         → packages/ml-frontend/src/Reward/calculateReward.ts
```

#### Entry Points
```
packages/ml/
  ├─ appo.html                  → packages/ml-frontend/index.html (объединить)
  └─ appo.ts                    → packages/ml-frontend/src/index.ts (адаптировать)
```

**Что НЕ переносим (остается в ml-backend):**
- `src/PPO/Learner/` - обучение остается на backend
- `src/PPO/LearnerPolicyWorker.ts` - worker обучения
- `src/PPO/LearnerValueWorker.ts` - worker обучения
- `src/PPO/train.ts` - V-trace и логика обучения

### 3. Адаптация под ml-backend инфраструктуру
**Цель:** Интегрировать с Supabase и Ably, заменить старую коммуникацию

**Важно:** Следуем подходу ml-backend - простые функции и RxJS каналы, без ООП оверинжиниринга.

#### 3.1 Загрузка моделей из Supabase
**Файл:** `src/Models/supabaseStorage.ts` (по аналогии с ml-backend)

**Подход (как в ml-backend):**
```typescript
// Простой клиент через замыкание
const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL || '';
const SUPABASE_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY || '';
let supabase: SupabaseClient | null = null;

function getSupabaseClient(): SupabaseClient | null {
    if (!supabase && SUPABASE_URL && SUPABASE_KEY) {
        supabase = createClient(SUPABASE_URL, SUPABASE_KEY);
    }
    return supabase;
}

// Простая функция загрузки модели
export async function loadModelFromSupabase(
    modelName: string, 
    version: number
): Promise<tf.LayersModel> {
    const client = getSupabaseClient();
    const url = getModelPublicUrl(modelName, version);
    return tf.loadLayersModel(url);
}

export function getModelPublicUrl(modelName: string, version: number): string {
    const client = getSupabaseClient();
    const { data } = client.storage
        .from(SUPABASE_BUCKET)
        .getPublicUrl(`v${version}-${modelName}/model.json`);
    return data.publicUrl;
}
```

**Задачи:**
- [ ] Реализовать `loadModelFromSupabase()` - простая функция загрузки
- [ ] Реализовать `getModelPublicUrl()` - получение URL модели
- [ ] Реализовать `checkLatestVersion()` - опрос bucket для новых версий

**Изменения в `Transfer.ts`:**
- Убрать сохранение моделей (это только на backend)
- Оставить только загрузку через `loadModelFromSupabase()`

#### 3.2 Отправка опыта через Ably
**Файл:** `src/channels.ts` (по аналогии с ml-backend/globalChannels.ts)

**Подход (как в ml-backend):**
```typescript
// Простая обертка над Ably через RxJS
import Ably from 'ably';
import { Observable, Subject } from 'rxjs';

const ably = new Ably.Realtime(import.meta.env.ABLY_API_KEY);

function createAblyChannel<T>(channelName: string) {
    const channel = ably.channels.get(channelName);
    const subject = new Subject<T>();

    channel.subscribe((message) => {
        subject.next(message.data as T);
    });

    return {
        obs: subject.asObservable(),
        emit: (data: T) => channel.publish('message', data)
    };
}

// Каналы для коммуникации с backend
export const episodeSampleChannel = createAblyChannel<EpisodeSample>('ml:experience.v1');
export const queueSizeChannel = createAblyChannel<number>('ml:queue-size.v1');
```

**Задачи:**
- [ ] Создать `createAblyChannel()` - обертка над Ably с RxJS Subject
- [ ] Создать `episodeSampleChannel` - для отправки опыта
- [ ] Создать `queueSizeChannel` - для мониторинга очереди backend
- [ ] Убрать старые BroadcastChannel из packages/ml

**Без классов `AblyClient` или `CommunicationManager`** - используем простые каналы и функции.

#### 3.3 Визуализация сбора опыта
**Файл:** `src/Visualization/GameRenderer.ts` (адаптировать из VisTest)

**Задачи:**
- [ ] Перенести `VisTestEpisodeManager`

### Phase 1: Перенос и базовая интеграция
- [ ] Весь код Actor/VisTest перенесен из packages/ml
- [ ] Код компилируется без ошибок
- [ ] Приложение запускается в браузере
- [ ] Базовый UI отображается

### Phase 2: Supabase интеграция
- [ ] загружаются модели из Supabase

### Phase 3: Ably интеграция
- [ ] Отправка батчей в ml:experience.v1 работает
- [ ] подписка на queue-size

### Phase 4: Визуализация
- [ ] Visual mode работает с рендерингом
- [ ] Можно включить/выключить режим рендеринга

## Технические детали

### Различия packages/ml vs ml-frontend

| Аспект | packages/ml (старый) | ml-frontend (новый) |
|--------|---------------------|---------------------|
| Роль | Actor + Learner в одном пакете | Только Actor (генерация опыта) |
| Workers | BroadcastChannel между воркерами | Ably для отправки данных на backend |
| Модели | Сохранение локально | Загрузка из Supabase |
| Обучение | Локальное (PPO/Learner) | На backend (ml-backend) |
| Визуализация | VisTest как опция | Основной режим работы |
| Каналы | Локальные BroadcastChannel | Ably Realtime для внешней связи |

### Переменные окружения

```env
# Supabase (для загрузки моделей)
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your_anon_key
VITE_SUPABASE_BUCKET=models

# Ably (для отправки опыта)
ABLY_API_KEY=your_ably_api_key
```


## Быстрый старт

```bash
# 1. Установить зависимости
npm install

# 2. Настроить окружение
cp .env.example .env
# Отредактировать .env с вашими credentials

# 3. Запустить dev сервер
npm run dev

# 4. Открыть браузер
open http://localhost:3001
```

## Следующие шаги

1. **Изучить packages/ml** - понять текущую реализацию Actor/VisTest
2. **Перенести Actor код** - EpisodeManager, VisTest, Worker, Reward
3. **Создать channels.ts** - простые RxJS каналы для Ably (по примеру ml-backend/globalChannels.ts)
4. **Создать supabaseStorage.ts** - простые функции загрузки моделей (по примеру ml-backend/supabaseStorage.ts)
5. **Адаптировать index.ts** - главный цикл: загрузка моделей → запуск актора → отправка опыта
6. **Интегрировать визуализацию** - перенести рендеринг из VisTest
7. **Тестирование** - проверить полный цикл Actor → Backend

## Архитектурные принципы (как в ml-backend)

**DO:**
- ✅ Простые функции для работы с Supabase/Ably
- ✅ RxJS Subject/Observable для каналов
- ✅ Функциональный стиль, минимум абстракций
- ✅ Переиспользовать подход из ml-backend

**DON'T:**
- ❌ Классы `ModelManager`, `AblyClient`, `CommunicationManager`
- ❌ IndexedDB кэширование (пока не нужно)
- ❌ Сложные абстракции и интерфейсы
- ❌ Переусложнение - следуем духу pet project

