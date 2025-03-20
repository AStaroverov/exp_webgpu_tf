import { TasksManager } from './TasksManager';

class MacroTasks extends TasksManager {
    constructor(delay = 2) {
        super((fn) => {
            const id = globalThis.setInterval(fn, delay);
            return () => globalThis.clearInterval(id);
        });
    }
}

export const macroTasks = new MacroTasks();
