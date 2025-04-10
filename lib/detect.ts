export const isBrowser = typeof globalThis.document != 'undefined' ||
    // @ts-ignore
    (typeof self !== 'undefined' && typeof self.importScripts === 'function');
export const isNode = typeof process != 'undefined' && process.versions != null && process.versions.node != null;