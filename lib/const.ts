export const EMPTY_STRING = '';
export const EMPTY_OBJECT = Object.freeze(Object.create(null));
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const EMPTY_ARRAY = Object.freeze([]) as unknown as any[];
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const EMPTY_MAP: ReadonlyMap<any, any> = new Map();
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const EMPTY_SET: ReadonlySet<any> = new Set();
