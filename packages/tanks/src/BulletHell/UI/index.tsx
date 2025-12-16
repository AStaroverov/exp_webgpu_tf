import './index.css';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import { BaseScreen } from './Widgets/BaseScreen.tsx';
import { BulletHellStateEffects } from '../State/GameState.ts';
import { HeroUIProvider } from '@heroui/react';
import { upsertModels } from '../../../../ml/src/Models/Trained/restore.ts';
import { initTensorFlow } from '../../../../ml-common/initTensorFlow.ts';

import '../../../../ml/src/Models/Layers';

await initTensorFlow('wasm');
await upsertModels('/assets/models/v1');

createRoot(document.getElementById('ui')!).render(
    <StrictMode>
        <HeroUIProvider className="w-full h-full">
            <BaseScreen className="w-full h-full"/>
        </HeroUIProvider>
    </StrictMode>,
);

BulletHellStateEffects().subscribe();
