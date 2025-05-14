import './index.css';
import React from 'react';
import ReactDOM from 'react-dom/client';

import { BaseScreen } from './Components/BaseScreen.tsx';

ReactDOM.createRoot(document.getElementById('ui')!).render(
    <React.StrictMode>
        <BaseScreen className="w-full h-full"/>
    </React.StrictMode>,
);
