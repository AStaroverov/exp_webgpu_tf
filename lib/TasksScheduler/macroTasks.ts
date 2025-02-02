import { TasksManager } from './TasksManager';

class MacroTasks extends TasksManager {
    constructor(delay = 8) {
        super((fn) => {
            const id = window.setInterval(fn, delay);
            return () => window.clearInterval(id);
        });
    }
}

export const macroTasks = new MacroTasks();
