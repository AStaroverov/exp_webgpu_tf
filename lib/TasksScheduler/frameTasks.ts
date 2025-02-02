import { TasksManager } from './TasksManager';

class FrameTasks extends TasksManager {
    constructor() {
        super((fn) => {
            let id = requestAnimationFrame(function ticker() {
                fn();
                id = requestAnimationFrame(ticker);
            });

            return () => cancelAnimationFrame(id);
        });
    }

    protected getDelta(): number {
        // Delta between frames we measure in count, not in milliseconds
        return 1;
    }
}

export const frameTasks = new FrameTasks();
