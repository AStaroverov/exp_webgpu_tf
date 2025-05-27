import { Select, SelectItem } from '../Components/Selector.tsx';

const pilots = [
    { key: 0, label: 'Ranger' },
];

export function PilotSelector({ className }: { className?: string, tankEid?: number }) {
    return (
        <div className={ `${ className } gap-2` }>
            <Select
                className="max-w-xs"
                label="Tank type"
                value={ 0 }
            >
                { pilots.map((item) => <SelectItem key={ item.key }>{ item.label }</SelectItem>) }
            </Select>
        </div>
    );
}
