import { Button, ButtonGroup } from '../Components/Button.tsx';
import { mapAddTank } from '../State/Game/playerMethods.ts';
import { TankType } from '../../Game/ECS/Components/Tank.ts';

export function AddTankButton({ className }: { className?: string }) {
    return (
        <ButtonGroup className={ className }>
            <Button color="success" onClick={ mapAddTank[TankType.Light] }>+ Light</Button>
            <Button color="warning" onClick={ mapAddTank[TankType.Medium] }>+ Medium</Button>
            <Button color="danger" onClick={ mapAddTank[TankType.Heavy] }>+ Heavy</Button>
        </ButtonGroup>
    );
}