import { Observable } from 'rxjs';
import { frameTasks } from '../TasksScheduler/frameTasks.ts';

export function frameInterval(delay: number = 1) {
    return new Observable<number>((subscriber) => {
        let time = Date.now();
        return frameTasks.addInterval(() => {
            const t = Date.now();
            const d = t - time;
            time = t;
            subscriber.next(d);
        }, delay);
    });
}
