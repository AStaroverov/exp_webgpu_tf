export function throwingError(message: string): never {
    throw new Error(message);
}