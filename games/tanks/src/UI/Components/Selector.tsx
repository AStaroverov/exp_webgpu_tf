import { SelectItem as _SelectItem } from '@heroui/select';
import { ReactElement } from 'react';
import { ListboxItemBaseProps } from '@heroui/listbox/dist/base/listbox-item-base';

export * from '@heroui/select';

export const SelectItem: <T extends object>(props: Omit<ListboxItemBaseProps<T>, 'value'>) => ReactElement = _SelectItem as any;
