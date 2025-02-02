import { LinkedMap, TLinkedMap } from './index';

const IDS = Array(10)
    .fill(0)
    .map((_, i) => String(i));
const VALUES = Array(10)
    .fill(0)
    .map((_, i) => 'v:' + i);

describe('LinkedMap tests', () => {
    let linkedMap: TLinkedMap<string, string>;

    beforeEach(() => {
        linkedMap = LinkedMap.create<string, string>();

        LinkedMap.insertLast(linkedMap, IDS[0], VALUES[0]);
        LinkedMap.insertLast(linkedMap, IDS[1], VALUES[1]);
        LinkedMap.insertLast(linkedMap, IDS[2], VALUES[2]);
    });

    describe('insertFirst', () => {
        it('add a node at the beginning of the list', () => {
            LinkedMap.insertFirst(linkedMap, IDS[0], VALUES[0]);
            LinkedMap.insertFirst(linkedMap, IDS[1], VALUES[1]);

            expect(linkedMap.map[IDS[1]].value).toEqual(VALUES[1]);
            expect(linkedMap.map[IDS[1]].nextId).toEqual(IDS[0]);
            expect(linkedMap.map[IDS[1]].prevId).toEqual(null);

            expect(linkedMap.map[IDS[0]].value).toEqual(VALUES[0]);
            expect(linkedMap.map[IDS[0]].prevId).toEqual(IDS[1]);
        });
    });

    describe('insertLast', () => {
        it('add a node to the end of the list', () => {
            const node0 = LinkedMap.getNode(linkedMap, IDS[0])!;
            expect(node0.value).toEqual(VALUES[0]);
            expect(node0.prevId).toEqual(null);
            expect(node0.nextId).toEqual(IDS[1]);

            const node2 = LinkedMap.getNode(linkedMap, IDS[2])!;
            expect(node2.value).toEqual(VALUES[2]);
            expect(node2.prevId).toEqual(IDS[1]);
            expect(node2.nextId).toEqual(null);
        });
    });

    describe('insertAt(position, value)', () => {
        it('add a node at a specific position', () => {
            LinkedMap.insertAt(linkedMap, 1, IDS[3], VALUES[3]);

            expect(linkedMap.map[IDS[3]].value).toEqual(VALUES[3]);
            expect(linkedMap.map[IDS[3]].nextId).toEqual(IDS[1]);
            expect(linkedMap.map[IDS[3]].prevId).toEqual(IDS[0]);

            LinkedMap.insertAt(linkedMap, 0, IDS[4], VALUES[4]);

            expect(linkedMap.map[IDS[4]].value).toEqual(VALUES[4]);
            expect(linkedMap.map[IDS[4]].nextId).toEqual(IDS[0]);
            expect(linkedMap.map[IDS[4]].prevId).toEqual(null);
        });

        it('throws error position is not a valid number', () => {
            expect(() => LinkedMap.insertAt(linkedMap, 999, IDS[0], VALUES[0])).toThrow(Error);
        });
    });

    describe('size', () => {
        it('get the size of nodes in the list', () => {
            expect(linkedMap.size).toEqual(3);
        });
    });

    describe('removeFirst', () => {
        it('remove the first node', () => {
            const removed = LinkedMap.removeFirst(linkedMap);
            expect(removed!.id).toEqual(IDS[0]);
            expect(linkedMap.size).toEqual(2);
            expect(linkedMap.headId).toEqual(IDS[1]);

            const head = LinkedMap.getNode(linkedMap, linkedMap.headId!);
            expect(head!.id).toEqual(IDS[1]);
            expect(head!.prevId).toEqual(null);
        });

        it('remove first node with a single node list', () => {
            const linkedMap = LinkedMap.create<string, string>();
            LinkedMap.insertFirst(linkedMap, '1', 'test');

            const removed = LinkedMap.removeFirst(linkedMap);
            expect(removed!.id).toEqual('1');
            expect(linkedMap.size).toEqual(0);
            expect(linkedMap.headId).toEqual(null);
            expect(linkedMap.tailId).toEqual(null);
        });
    });

    describe('removeLast', () => {
        it('remove the last node', () => {
            const removed = LinkedMap.removeLast(linkedMap);
            expect(removed!.id).toEqual(IDS[2]);
            expect(linkedMap.size).toEqual(2);
            expect(linkedMap.tailId).toEqual(IDS[1]);

            const tail = LinkedMap.getNode(linkedMap, linkedMap.tailId!);
            expect(tail!.id).toEqual(IDS[1]);
            expect(tail!.nextId).toEqual(null);
        });

        it('remove last node with a single node list', () => {
            const linkedMap = LinkedMap.create<string, string>();
            LinkedMap.insertFirst(linkedMap, '1', 'test');

            const removed = LinkedMap.removeLast(linkedMap);
            expect(removed!.id).toEqual('1');
            expect(linkedMap.size).toEqual(0);
            expect(linkedMap.headId).toEqual(null);
            expect(linkedMap.tailId).toEqual(null);
        });
    });

    describe('at', () => {
        it('get node by index', () => {
            expect(LinkedMap.getNodeAt(linkedMap, 1)?.id).toEqual(IDS[1]);
            expect(LinkedMap.getNodeAt(linkedMap, -1)?.id).toEqual(IDS[2]);
        });

        it('get value by index', () => {
            expect(LinkedMap.getValueAt(linkedMap, 1)).toEqual(VALUES[1]);
            expect(LinkedMap.getValueAt(linkedMap, -1)).toEqual(VALUES[2]);
        });
    });

    describe('removeAt', () => {
        it('remove a node', () => {
            const removeIndex = 1;
            const removed = LinkedMap.removeAt(linkedMap, removeIndex);

            expect(removed!.id).toEqual(IDS[removeIndex]);
            expect(linkedMap.size).toEqual(2);

            const node0 = LinkedMap.getNode(linkedMap, IDS[0]);
            const node2 = LinkedMap.getNode(linkedMap, IDS[2]);

            expect(node0!.nextId).toEqual(node2!.id);
            expect(node2!.prevId).toEqual(node0!.id);
        });

        it('does nothing if position is not valid', () => {
            expect(LinkedMap.removeAt(linkedMap, 999)).toEqual(undefined);
        });
    });

    describe('removeById', () => {
        it('remove a node', () => {
            const removeId = IDS[1];
            const removed = LinkedMap.removeById(linkedMap, removeId);

            expect(removed!.id).toEqual(removeId);
            expect(linkedMap.size).toEqual(2);

            const node0 = LinkedMap.getNode(linkedMap, IDS[0]);
            const node2 = LinkedMap.getNode(linkedMap, IDS[2]);

            expect(node0!.nextId).toEqual(node2!.id);
            expect(node2!.prevId).toEqual(node0!.id);
        });

        it('does nothing if position is not valid', () => {
            expect(LinkedMap.removeById(linkedMap, 'random')).toEqual(undefined);
        });
    });

    describe('removeByValue', () => {
        it('remove a node', () => {
            const removeValue = VALUES[1];
            const removed = LinkedMap.removeByValue(linkedMap, removeValue);

            expect(removed!.value).toEqual(removeValue);
            expect(linkedMap.size).toEqual(2);

            const node0 = LinkedMap.getNode(linkedMap, IDS[0]);
            const node2 = LinkedMap.getNode(linkedMap, IDS[2]);

            expect(node0!.nextId).toEqual(node2!.id);
            expect(node2!.prevId).toEqual(node0!.id);
        });

        it('does nothing if position is not valid', () => {
            expect(LinkedMap.removeByValue(linkedMap, 'random')).toEqual(undefined);
        });
    });

    describe('forEach', () => {
        it('traverse the linked list', () => {
            LinkedMap.forEach(linkedMap, (node, index) => {
                expect(node.id).toEqual(IDS[index]);
                expect(node.value).toEqual(VALUES[index]);
            });
        });
    });

    describe('find', () => {
        it('find node', () => {
            const node = LinkedMap.find(linkedMap, (node) => node.id === IDS[2]);
            expect(node!.id).toEqual(IDS[2]);
        });
    });

    describe('findIndex', () => {
        it('findIndex node', () => {
            const index0 = LinkedMap.findIndex(linkedMap, (node) => node.id === IDS[0]);
            expect(index0).toEqual(0);

            const index1 = LinkedMap.findIndex(linkedMap, (node) => node.id === IDS[1]);
            expect(index1).toEqual(1);

            const index_1 = LinkedMap.findIndex(linkedMap, (node) => node.id === '');
            expect(index_1).toEqual(-1);
        });
    });
});
