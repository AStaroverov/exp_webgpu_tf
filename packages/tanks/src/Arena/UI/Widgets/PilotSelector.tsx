import { ChangeEvent, useCallback } from 'react';
import { changeTankType } from '../../State/Game/gameMethods.ts';
import { Select, SelectItem } from '../Components/Selector.tsx';

const pilots = [
    { key: 0, label: 'Player' },
    { key: 1, label: 'Pilot v1' },
];

export function PilotSelector({ className, tankEid, slot }: { className?: string, tankEid: number, slot: number }) {
    const handleChangePilot = useCallback((event: ChangeEvent<{ value: string }>) => {
        changeTankType(tankEid, slot, Number(event.target.value));
    }, [tankEid, slot]);
    return (
        <div className={ `${ className } gap-2` }>
            <Select
                className="max-w-xs"
                label="Pilot"
                defaultSelectedKeys={ [0] }
                onChange={ handleChangePilot }
            >
                { pilots.map((item) => <SelectItem key={ item.key }>{ item.label }</SelectItem>) }
            </Select>
        </div>
    );
}
