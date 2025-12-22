import './index.css';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import { BaseScreen } from './Widgets/BaseScreen.tsx';
import { GameMenuEffects } from './Effects/GameMenu.ts';
import { HeroUIProvider } from '@heroui/react';
import { initGameInitEffect } from '../State/Game/effects/gameInitEffect.ts';

createRoot(document.getElementById('ui')!).render(
    <StrictMode>
        <HeroUIProvider className="w-full h-full">
            <BaseScreen className="w-full h-full"/>
        </HeroUIProvider>
    </StrictMode>,
);

initGameInitEffect().subscribe();
GameMenuEffects().subscribe();