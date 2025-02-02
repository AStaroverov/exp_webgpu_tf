// Based on https://github.com/datastructures-js/linked-list/blob/master/src/doublyLinkedList.js

import { Node, TNode } from './node';

export type TLinkedMap<ID extends string | number, V> = {
    size: 0;
    map: Record<ID, TNode<ID, V>>;
    headId: null | ID;
    tailId: null | ID;
};

function create<ID extends string | number, V>(): TLinkedMap<ID, V> {
    return {
        size: 0,
        headId: null,
        tailId: null,
        map: {} as Record<ID, TNode<ID, V>>,
    };
}

function setHeadId<ID extends string | number, V>(self: TLinkedMap<ID, V>, id: null | ID): void {
    self.headId = id;
}

function setTailId<ID extends string | number, V>(self: TLinkedMap<ID, V>, id: null | ID): void {
    self.tailId = id;
}

function createNode<ID extends string | number, V>(
    self: TLinkedMap<ID, V>,
    id: ID,
    value: V,
    prev?: ID,
    next?: ID,
): TNode<ID, V> {
    const n = Node.create(id, value, prev, next);
    self.map[n.id] = n;
    self.size += 1;
    return n;
}

function deleteNode<ID extends string | number, V>(self: TLinkedMap<ID, V>, id: ID): void {
    delete self.map[id];
    self.size -= 1;
}

function hasNode<ID extends string | number, V>(self: TLinkedMap<ID, V>, id: ID): boolean {
    return id in self.map;
}

function getNode<ID extends string | number, V>(
    self: TLinkedMap<ID, V>,
    id: ID,
): undefined | TNode<ID, V> {
    return self.map[id];
}

function setNodeValue<ID extends string | number, V>(
    self: TLinkedMap<ID, V>,
    id: ID,
    value: V,
): void {
    const node = getNode(self, id);
    node && Node.setValue(node, value);
}

function getNodeValue<ID extends string | number, V>(
    self: TLinkedMap<ID, V>,
    id: ID,
): undefined | V {
    return getNode(self, id)?.value;
}

function unsafeGetNode<ID extends string | number, V>(
    self: TLinkedMap<ID, V>,
    id: ID,
): TNode<ID, V> {
    return self.map[id]!;
}

function clear<ID extends string | number, V>(list: TLinkedMap<ID, V>): void {
    list.size = 0;
    list.map = {} as Record<ID, TNode<ID, V>>;
    list.headId = null;
    list.tailId = null;
}

function insertFirst<ID extends string | number, V>(
    self: TLinkedMap<ID, V>,
    id: ID,
    value: V,
): TLinkedMap<ID, V> {
    const newNode = createNode(self, id, value);

    if (self.headId !== null && self.tailId !== null) {
        Node.setPrevId(unsafeGetNode(self, self.headId), newNode.id);
        Node.setNextId(newNode, self.headId);
        setHeadId(self, newNode.id);
    } else {
        setHeadId(self, newNode.id);
        setTailId(self, newNode.id);
    }

    return self;
}

function insertLast<ID extends string | number, V>(
    self: TLinkedMap<ID, V>,
    id: ID,
    value: V,
): TLinkedMap<ID, V> {
    const newNode = createNode(self, id, value);

    if (self.headId !== null && self.tailId !== null) {
        Node.setPrevId(newNode, self.tailId);
        Node.setNextId(unsafeGetNode(self, self.tailId), newNode.id);
        setTailId(self, newNode.id);
    } else {
        setHeadId(self, newNode.id);
        setTailId(self, newNode.id);
    }

    return self;
}

function getNodeAt<ID extends string | number, V>(
    self: TLinkedMap<ID, V>,
    position: number,
): undefined | TNode<ID, V> {
    position = position < 0 ? self.size + position : position;

    if (position >= self.size) {
        return undefined;
    }

    if (position === 0) {
        return unsafeGetNode(self, self.headId!);
    }

    if (position === self.size - 1) {
        return unsafeGetNode(self, self.tailId!);
    }

    let currentPosition = 1;
    let currentId = unsafeGetNode(self, self.headId!).nextId!;

    while (currentPosition < position) {
        currentPosition += 1;
        currentId = unsafeGetNode(self, currentId!).nextId!;
    }

    return unsafeGetNode(self, currentId);
}

function getValueAt<ID extends string | number, V>(
    self: TLinkedMap<ID, V>,
    position: number,
): undefined | V {
    return getNodeAt(self, position)?.value;
}

function insertAt<ID extends string | number, V>(
    self: TLinkedMap<ID, V>,
    position: number,
    id: ID,
    value: V,
): TLinkedMap<ID, V> {
    if (position < 0 || position > self.size) {
        throw new Error('insertAt expects a position num <= linked list size');
    }

    if (position === 0) {
        return insertFirst(self, id, value);
    }

    if (position === self.size) {
        return insertLast(self, id, value);
    }

    let currentPosition = 1;
    let prevId = self.headId;

    while (currentPosition < position) {
        currentPosition += 1;
        prevId = prevId ? self.map[prevId]?.nextId ?? null : null;
    }

    const newNode = createNode(self, id, value);
    const prev = prevId ? unsafeGetNode(self, prevId) : null;
    const prevNext = prev?.nextId ? unsafeGetNode(self, prev.nextId) : null;

    Node.setNextId(newNode, prevNext?.id ?? null);
    Node.setPrevId(newNode, prevId);

    Node.setPrevId(unsafeGetNode(self, newNode.nextId!), newNode.id);
    Node.setNextId(unsafeGetNode(self, newNode.prevId!), newNode.id);

    return self;
}

function removeFirst<ID extends string | number, V>(
    self: TLinkedMap<ID, V>,
): undefined | TNode<ID, V> {
    if (self.headId === null) return undefined;

    const removedNodeId = self.headId;
    const removedNode = unsafeGetNode(self, self.headId);

    if (removedNode.nextId !== null) {
        setHeadId(self, removedNode.nextId);
        Node.setPrevId(unsafeGetNode(self, self.headId), null);
    } else {
        setHeadId(self, null);
        setTailId(self, null);
    }

    deleteNode(self, removedNodeId);

    return removedNode;
}

function removeLast<ID extends string | number, V>(
    self: TLinkedMap<ID, V>,
): undefined | TNode<ID, V> {
    if (self.tailId === null) return undefined;

    const removedNodeId = self.tailId;
    const removedNode = unsafeGetNode(self, self.tailId);

    if (removedNode.prevId !== null) {
        setTailId(self, removedNode.prevId);
        Node.setNextId(unsafeGetNode(self, self.tailId), null);
    } else {
        setHeadId(self, null);
        setTailId(self, null);
    }

    deleteNode(self, removedNodeId);

    return removedNode;
}

function removeById<ID extends string | number, V>(
    self: TLinkedMap<ID, V>,
    id: ID,
): undefined | TNode<ID, V> {
    const node = getNode(self, id);

    if (node === undefined) {
        return undefined;
    }

    if (node.prevId === null) {
        return removeFirst(self);
    }

    if (node.nextId === null) {
        return removeLast(self);
    }

    Node.setNextId(unsafeGetNode(self, node.prevId), node.nextId);
    Node.setPrevId(unsafeGetNode(self, node.nextId), node.prevId);

    deleteNode(self, node.id);

    return node;
}

function removeAt<ID extends string | number, V>(
    self: TLinkedMap<ID, V>,
    position: number,
): undefined | TNode<ID, V> {
    if (position === 0) {
        return removeFirst(self);
    }

    if (position === self.size - 1) {
        return removeLast(self);
    }

    const node = getNodeAt(self, position);

    return node === undefined ? undefined : removeById(self, node.id);
}

function removeByValue<ID extends string | number, V>(
    self: TLinkedMap<ID, V>,
    value: V,
): undefined | TNode<ID, V> {
    const index = findIndex(self, (n) => n.value === value);
    return index === -1 ? undefined : removeAt(self, index);
}

function find<ID extends string | number, V>(
    self: TLinkedMap<ID, V>,
    matcher: (node: TNode<ID, V>) => boolean,
): undefined | TNode<ID, V> {
    let current = self.headId === null ? null : unsafeGetNode(self, self.headId);

    while (current !== null) {
        if (matcher(current)) return current;
        current = current.nextId === null ? null : unsafeGetNode(self, current.nextId);
    }

    return undefined;
}

function findIndex<ID extends string | number, V>(
    self: TLinkedMap<ID, V>,
    matcher: (node: TNode<ID, V>) => boolean,
): number {
    let index = 0;
    let current = self.headId === null ? null : unsafeGetNode(self, self.headId);

    while (current !== null) {
        if (matcher(current)) return index;
        index += 1;
        current = current.nextId === null ? null : unsafeGetNode(self, current.nextId);
    }

    return -1;
}

function forEach<ID extends string | number, V>(
    self: TLinkedMap<ID, V>,
    callback: (node: TNode<ID, V>, index: number) => unknown,
): void {
    let id = self.headId;
    let node = undefined;
    let index = 0;
    const ids = new Array(self.size);

    while (id !== null) {
        ids[index++] = id;
        id = unsafeGetNode(self, id).nextId;
    }

    for (let i = 0; i < ids.length; i++) {
        node = getNode(self, ids[i]);
        if (node !== undefined) {
            callback(node, i);
        }
    }
}

export const LinkedMap = {
    create,
    clear,
    hasNode,
    getNode,
    getNodeValue,
    setNodeValue,
    insertFirst,
    insertLast,
    getNodeAt,
    getValueAt,
    insertAt,
    removeFirst,
    removeLast,
    removeAt,
    removeById,
    removeByValue,
    forEach,
    find,
    findIndex,
};
