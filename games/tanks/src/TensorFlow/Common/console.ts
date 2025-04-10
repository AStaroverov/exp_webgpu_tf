type ConsoleMethod = (...args: any[]) => void;
type ConsoleMethods = 'log' | 'warn' | 'error' | 'info' | 'debug';

// Интерфейс для хранения оригинальных методов
interface OriginalConsoleMethods {
    [key: string]: ConsoleMethod;
}

// Сохраняем оригинальные методы консоли
const originalConsole: OriginalConsoleMethods = {
    log: console.log,
    warn: console.warn,
    error: console.error,
    info: console.info,
    debug: console.debug,
};

/**
 * Альтернативный вариант с явным отображением стека вызовов
 * @param prefix Строка, используемая как префикс
 */
export const setConsolePrefix = (prefix: string): void => {
    // Перегрузка для всех методов в цикле
    (Object.keys(originalConsole) as ConsoleMethods[]).forEach((method) => {
        console[method] = function (...args: any[]): void {
            const stack: { stack?: string } = {};

            // Захватываем текущий стек вызовов
            Error.captureStackTrace(stack);

            // Извлекаем настоящую позицию вызова
            const stackLines = stack.stack?.split('\n') || [];
            const callerInfo = stackLines.length > 2 ? ' ' + stackLines[2].trim() : '';

            // if (method === 'error' || method === 'warn' || devtools.isOpen) {
            // Вызываем оригинальный метод с префиксом
            originalConsole[method].apply(console, [prefix, ...args, callerInfo]);
            // }
        };
    });
};
