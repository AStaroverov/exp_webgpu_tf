import 'devtools-detect';
import { Orientation } from 'devtools-detect';

export const devtoolsChannel = new BroadcastChannel('devtools-detect');

type DevToolsState = {
    isOpen: boolean;
    orientation?: Orientation;
}

export const devtools: DevToolsState = {
    isOpen: false,
    orientation: undefined,
};

devtoolsChannel.onmessage = (event) => {
    const data = event.data as DevToolsState;
    devtools.isOpen = data.isOpen;
    devtools.orientation = data.orientation;
};

if (globalThis.document != null) {
    const message = new BroadcastChannel('devtools-detect');
    globalThis.addEventListener('devtoolschange', (event) => {
        message.postMessage((event as CustomEvent<DevToolsState>).detail);
    });
}

