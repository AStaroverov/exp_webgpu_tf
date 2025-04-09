import PouchDB from 'pouchdb';

export function createMemoryBatchDB<Batch>(name: string) {
    const db = new PouchDB(name);

    async function addMemoryBatch(batch: { version: number; memories: Batch }) {
        const id = Date.now().toString();
        return db.put({ _id: id, ...batch });
    }

    async function getMemoryBatchCount() {
        const result = await db.allDocs();
        return result.total_rows;
    }

    async function getMemoryBatchList() {
        const result = await db.allDocs({ include_docs: true });
        return result.rows.map(r => r.doc as unknown as { _id: string; version: number; memories: Batch });
    }

    async function clearMemoryBatchList() {
        const docs = await db.allDocs({ include_docs: true });
        const deletions = docs.rows.map(row => ({
            ...row.doc,
            _deleted: true,
        }));
        return db.bulkDocs(deletions);
    }

    async function extractMemoryBatchList() {
        const list = await getMemoryBatchList();
        await clearMemoryBatchList();
        return list;
    }

    return {
        getMemoryBatchCount,
        addMemoryBatch,
        getMemoryBatchList,
        clearMemoryBatchList,
        extractMemoryBatchList,
    };
}