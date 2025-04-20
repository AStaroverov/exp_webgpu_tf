export const isBrowser = typeof globalThis.document != 'undefined' ||
    // @ts-ignore
    (typeof self !== 'undefined' && typeof self.importScripts === 'function');
export const isNode = typeof process != 'undefined' && process.versions != null && process.versions.node != null;
export const isMac = isBrowser && navigator.platform.toUpperCase().indexOf('MAC') >= 0;

export const isTabThread = isBrowser && globalThis.document != null;