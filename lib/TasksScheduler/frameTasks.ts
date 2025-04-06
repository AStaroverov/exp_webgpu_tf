import { TasksManager } from './TasksManager';

class FrameTasks extends TasksManager {
    constructor() {
        const request = globalThis.requestAnimationFrame ?? global.setTimeout;
        const cancel = globalThis.cancelAnimationFrame ?? global.clearTimeout;
        super((fn) => {
            let id = request(function ticker() {
                fn();
                id = request(ticker);
            });

            return () => cancel(id);
        });
    }

    protected getDelta(): number {
        // Delta between frames we measure in count, not in milliseconds
        return 1;
    }
}

export const frameTasks = new FrameTasks();
