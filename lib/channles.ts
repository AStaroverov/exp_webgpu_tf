import { merge, Observable, Subject } from 'rxjs';

export function createChannel<Req, Res = unknown>(name: string) {
    const request = new Subject<Req>();
    const response = new Subject<Res>();
    const crossRequest = new BroadcastChannel(name + '-request');
    const crossResponse = new BroadcastChannel(name + '-response');

    const request$ = merge(
        request,
        new Observable<Req>((observer) => {
            crossRequest.onmessage = (event) => observer.next(event.data);
            return () => {
                crossRequest.onmessage = null;
                crossRequest.close();
            };
        }),
    );

    const response$ = merge(
        response,
        new Observable<Res>((observer) => {
            crossResponse.onmessage = (event) => observer.next(event.data);
            return () => {
                crossResponse.onmessage = null;
                crossResponse.close();
            };
        }),
    );

    return {
        emit: (data: Req) => {
            request.next(data);
            crossRequest.postMessage(data);
        },
        obs: request$,
        request: (data: Req) => {
            request.next(data);
            crossRequest.postMessage(data);
            return response$;
        },
        response: (cb: (data: Req) => Res) => {
            const sub = request$.subscribe((data) => {
                const res = cb(data);
                response.next(res);
                crossResponse.postMessage(res);
            });

            return () => sub.unsubscribe();
        },
    };
}