import { Select, SelectItem } from '../Components/Selector.tsx';
import { useCallback, useEffect, useState } from 'react';
import { changeTankPilotBySlot, getPilotTypeBySlot$ } from '../State/Game/pilotsMethods.ts';
import { ValueOf } from '../../../../../lib/Types';
import { useObservable } from '../../../../../lib/React/useSyncObservable.ts';
import { PilotType } from '../../Pilots/Components/Pilot.ts';
import { SharedSelection } from '@heroui/system';
import { Key } from '@react-types/shared';

const pilots = [
    { key: PilotType.Player, label: 'Player' },
    { key: PilotType.Agent31, label: 'Pilot v31' },
    { key: PilotType.Agent32, label: 'Pilot v32' },
];

export function PilotSelector({ className, slot }: { className?: string, slot: number }) {
    const [value, setValue] = useState<Selection>();
    const selectedKey = useObservable(getPilotTypeBySlot$(slot));
    const handleChangePilot = useCallback((event: SharedSelection) => {
        changeTankPilotBySlot(slot, Number(event.currentKey) as ValueOf<typeof PilotType>);
    }, []);

    useEffect(() => {
        const v = selectedKey == null || selectedKey === 0
            ? undefined
            : String(selectedKey);
        setValue(new Selection(v ? [v] : undefined, v, v));
    }, [selectedKey]);

    return (
        <div className={ `${ className } gap-2` }>
            <Select
                className="max-w-xs"
                label="Pilot"
                selectionMode="single"
                // defaultSelectedKeys={ [selectedKey ?? 0] }
                selectedKeys={ value }
                onSelectionChange={ handleChangePilot }
            >
                { pilots.map((item) => (
                    <SelectItem key={ item.key }>
                        { item.label }
                    </SelectItem>
                )) }
            </Select>
        </div>
    );
}

export class Selection extends Set<Key> {
    anchorKey: Key | null;
    currentKey: Key | null;

    constructor(keys?: Iterable<Key> | Selection, anchorKey?: Key | null, currentKey?: Key | null) {
        super(keys);
        if (keys instanceof Selection) {
            this.anchorKey = anchorKey ?? keys.anchorKey;
            this.currentKey = currentKey ?? keys.currentKey;
        } else {
            this.anchorKey = anchorKey ?? null;
            this.currentKey = currentKey ?? null;
        }
    }
}
