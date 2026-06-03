import { createChannel } from '../../../lib/channles.ts';
import { CurriculumState } from './curriculum/types.ts';

export const curriculumStateChannel = createChannel<CurriculumState>('curriculumStateChannel');
