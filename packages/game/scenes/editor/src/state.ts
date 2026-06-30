import { BehaviorSubject } from "rxjs";
import { ENTITIES } from "../../../src/Entities/registry.js";
import type { EditClip } from "../../../src/anim/editclip.js";

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

// The Animation selector carries a special "edit" value that enters the authoring pipeline.
export const EDIT = "edit";

export const selectedEntityId$ = persistentState<string>("viewer.entity", ENTITIES[0].id);
export const selectedAnimation$ = persistentState<string>("viewer.animation", "none");
export const selectedScale$ = persistentState<number>("viewer.scale", 1);
export const selectedEid$ = new BehaviorSubject<number>(-1);

// The clip being authored in the right-panel pipeline (in-session; bound to one entity type).
export const editClip$ = new BehaviorSubject<EditClip | null>(null);
