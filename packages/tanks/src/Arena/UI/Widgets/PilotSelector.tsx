import { Select, SelectItem } from '../Components/Selector.tsx';

const pilots = [
    { key: 0, label: 'Player' },
    { key: 31, label: 'Pilot v31' },
    { key: 32, label: 'Pilot v32' },
];

export function PilotSelector({ className, slot }: { className?: string, slot: number }) {
    // const handleChangePilot = useCallback(() => {
    //     changeTankType();
    // }, []);
    return (
        <div className={ `${ className } gap-2` }>
            <Select
                className="max-w-xs"
                label="Pilot"
                defaultSelectedKeys={ [0] }
                // onChange
            >
                { pilots.map((item) => <SelectItem key={ item.key }>{ item.label }</SelectItem>) }
            </Select>
        </div>
    );
}
