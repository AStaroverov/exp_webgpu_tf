import { Select, SelectItem } from '../Components/Selector.tsx';
import { TankType } from '../../../Game/ECS/Components/Tank.ts';
import { useObservable } from '../../../../../../lib/React/useSyncObservable.ts';
import { changeTankType, getTankType$ } from '../../State/Game/gameMethods.ts';
import { EMPTY } from 'rxjs';
import { ChangeEvent, useCallback } from 'react';
import { getTankEidBySlot$ } from '../../State/Game/playerMethods.ts';

const tankTypes = [
    { key: TankType.Light, label: 'Light' },
    { key: TankType.Medium, label: 'Medium' },
    { key: TankType.Heavy, label: 'Heavy' },
];

export function TankSelector({ className, slot }: { className?: string, slot: number, }) {
    const tankEid = useObservable(getTankEidBySlot$(slot));
    const tankType = useObservable(tankEid ? getTankType$(tankEid) : EMPTY);
    const handleChangeTankType = useCallback((event: ChangeEvent<{ value: string }>) => {
        if (tankEid == null) return;
        changeTankType(tankEid, slot, Number(event.target.value));
    }, [tankEid]);

    return (
        <div className={ `${ className } gap-2` }>
            <Select
                className="max-w-xs"
                label="Tank type"
                value={ tankType }
                onChange={ handleChangeTankType }
            >
                { tankTypes.map((item) => <SelectItem key={ item.key }>{ item.label }</SelectItem>) }
            </Select>
        </div>
    );
}
