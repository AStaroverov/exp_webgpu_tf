# Перенос клиентского кода из packages/ml

## Выполнено ✅

### Скопированы файлы (без изменений):

```
packages/ml → packages/ml-frontend

src/PPO/Actor/
  └─ EpisodeManager.ts              ✅ src/Actor/EpisodeManager.ts

src/PPO/VisTest/
  └─ VisTestEpisodeManager.ts       ✅ src/Actor/VisTest/VisTestEpisodeManager.ts

src/PPO/
  ├─ ActorWorker.ts                 ✅ src/Actor/ActorWorker.ts
  └─ channels.ts                    ✅ src/channels.ts

src/Reward/
  └─ calculateReward.ts             ✅ src/Reward/calculateReward.ts

appo.html                           ✅ actor.html
appo.ts                             ✅ actor.ts (адаптирован)
```

### Структура ml-frontend после переноса:

```
packages/ml-frontend/
├── actor.html                       # Entry HTML
├── actor.ts                         # Main entry point
├── index.html                       # Index page (создан ранее)
├── package.json
├── tsconfig.json
├── vite.config.ts
├── .env.example
│
└── src/
    ├── channels.ts                  # BroadcastChannel definitions
    ├── config.ts                    # Configuration
    ├── index.ts                     # Alternative entry point
    ├── vite-env.d.ts               # Vite types
    │
    ├── Actor/
    │   ├── EpisodeManager.ts        # Core episode management
    │   ├── ActorWorker.ts           # Worker для headless mode
    │   └── VisTest/
    │       └─ VisTestEpisodeManager.ts  # Visual episode manager
    │
    ├── Reward/
    │   └── calculateReward.ts       # Reward function
    │
    └── Models/ (пусто, будет заполнено позже)
```

## Следующие шаги

### Phase 1: Исправить импорты ⏳
Сейчас все файлы имеют ошибки импортов, т.к. ссылаются на:
- `../../../lib/` - общие утилиты
- `../../ml-common/` - общий пакет ml-common
- `../../tanks/` - игровая логика

**Нужно:**
1. Проверить что `packages/ml-common` доступен (это отдельный пакет)
2. Проверить что `lib/` доступен (это корневая lib)
3. Проверить что `packages/tanks` доступен
4. Возможно нужно настроить пути в tsconfig.json

### Phase 2: Адаптировать channels.ts
- Заменить `BroadcastChannel` на Ably для `episodeSampleChannel`
- Оставить локальные каналы если нужны
- Добавить `queueSizeChannel` подписку

### Phase 3: Добавить Supabase загрузку моделей
- Создать `src/Models/supabaseStorage.ts`
- Функции: `loadModelFromSupabase()`, `getModelPublicUrl()`
- Заменить в actor.ts загрузку моделей

### Phase 4: Проверить что все работает
- `npm install`
- `npm run dev`
- Открыть http://localhost:3001/actor.html

## Заметки

- Все файлы скопированы **БЕЗ ИЗМЕНЕНИЙ** как и планировалось
- Структура сохранена максимально близко к оригиналу
- Сейчас код не компилируется из-за импортов - это нормально, исправим на следующих шагах
- Следуем принципу: сначала переносим, потом адаптируем
