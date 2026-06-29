import { BehaviorSubject } from "rxjs";
import { ENTITIES } from "./Entities/registry.ts";

export function persistentState<T>(key: string, initial: T): BehaviorSubject<T> {
  let start = initial;
  const raw = localStorage.getItem(key);
  if (raw !== null) {
    try {
      start = JSON.parse(raw) as T;
    } catch {}
  }
  const subject = new BehaviorSubject<T>(start);
  subject.subscribe((v) => localStorage.setItem(key, JSON.stringify(v)));
  return subject;
}

export const selectedEntityId$ = persistentState<string>("viewer.entity", ENTITIES[0].id);
export const selectedEid$ = new BehaviorSubject<number>(-1);
