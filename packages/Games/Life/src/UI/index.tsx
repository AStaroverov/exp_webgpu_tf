import './index.css';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import { HeroUIProvider } from '@heroui/react';
import { initGameInitEffect } from '../Game/effects/gameInitEffect';
import { BaseScreen } from './Widgets/BaseScreen';

createRoot(document.getElementById('ui')!).render(
    <StrictMode>
        <HeroUIProvider className="w-full h-full">
            <BaseScreen className="w-full h-full"/>
        </HeroUIProvider>
    </StrictMode>,
);

initGameInitEffect().subscribe();