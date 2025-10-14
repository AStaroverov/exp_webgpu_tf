import { forceExitChannel } from './channels.ts';

globalThis.addEventListener("error", function (e) {
    console.error("Error occurred: ", e.message);
    forceExitChannel.postMessage(true);
    return false;
})

globalThis.addEventListener('unhandledrejection', function (e) {
    console.error("Unhandled rejection: ", e.reason);
    forceExitChannel.postMessage(true);
    return false;
})