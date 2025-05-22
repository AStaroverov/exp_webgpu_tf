import { Actions } from './applyActionsToTank.ts';
import { FlipMode, flipVec } from '../InputArrays/flipInputArrays.ts';

export function flipActions(a: Actions, m: FlipMode = 'none'): Actions {
    if (m === 'none') return a;

    a = a.slice() as Actions;

    [a[3], a[4]] = flipVec(m, a[3], a[4]);

    if (m !== 'xy') a[2] = -a[2];

    return a;
}