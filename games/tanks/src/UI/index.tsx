import './index.css';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import { BaseScreen } from './Widgets/BaseScreen.tsx';
import { GameMenuEffects } from './Effects/GameMenu.ts';
import { GameStateEffects } from './State/Game/GameState.ts';

createRoot(document.getElementById('ui')!).render(
    <StrictMode>
        <BaseScreen className="w-full h-full"/>
    </StrictMode>,
);

GameStateEffects().subscribe();
GameMenuEffects().subscribe();