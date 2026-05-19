import { macroTasks } from '../../../../lib/TasksScheduler/macroTasks.ts';

export async function patientAction<T>(action: () => T | Promise<T>, attempts: number = 100): Promise<T> {
    while (true) {
        attempts--;
        try {
            return await action();
        } catch (error) {
            if (attempts <= 0) {
                throw error;
            }

            await new Promise(resolve => macroTasks.addTimeout(resolve, 30));
        }
    }
}
