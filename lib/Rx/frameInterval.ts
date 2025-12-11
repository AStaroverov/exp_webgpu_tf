import { Observable } from 'rxjs';
import { frameTasks } from '../TasksScheduler/frameTasks.ts';

export function frameInterval(delayMs: number = 0) {
    return new Observable<number>((subscriber) => {
        let lastEmitTime = Date.now();
        let elapsed = 0;

        return frameTasks.addInterval(() => {
            const now = Date.now();
            elapsed += now - lastEmitTime;
            lastEmitTime = now;

            if (elapsed >= delayMs) {
                subscriber.next(elapsed);
                elapsed = 0;
            }
        }, 1);
    });
}
