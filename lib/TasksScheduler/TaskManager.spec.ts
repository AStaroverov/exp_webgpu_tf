import { TasksManager } from './TasksManager';

describe('TasksManager', () => {
    let originalNow: (typeof Date)['now'];
    const nowMock = jest.fn();

    beforeAll(() => {
        originalNow = Date.now;
        Date.now = nowMock;
    });

    afterAll(() => {
        Date.now = originalNow;
    });

    beforeEach(() => {
        nowMock.mockReturnValue(0);
    });

    it('Should register callback', () => {
        const ticker = jest.fn();
        new TasksManager(ticker);
        expect(ticker.mock.calls.length).toBe(1);
    });

    it('Should run single task', () => {
        const ticker = jest.fn();
        const taskManager = new TasksManager(ticker);

        const task = jest.fn();

        taskManager.addInterval(task, 500);

        nowMock.mockReturnValue(500);
        ticker.mock.calls[0][0]();

        expect(task.mock.calls.length).toBe(1);

        nowMock.mockReturnValue(1000);
        ticker.mock.calls[0][0]();

        expect(task.mock.calls.length).toBe(2);

        nowMock.mockReturnValue(5000);
        ticker.mock.calls[0][0]();

        expect(task.mock.calls.length).toBe(3);
    });

    it('Should run multiple task', () => {
        const ticker = jest.fn();
        const taskManager = new TasksManager(ticker);

        const task1 = jest.fn();
        const task2 = jest.fn();

        const removeTask1 = taskManager.addInterval(task1, 500);
        taskManager.addInterval(task2, 1000);

        nowMock.mockReturnValue(500);
        ticker.mock.calls[0][0]();

        expect(task1.mock.calls.length).toBe(1);
        expect(task2.mock.calls.length).toBe(0);

        nowMock.mockReturnValue(1000);
        ticker.mock.calls[0][0]();

        expect(task1.mock.calls.length).toBe(2);
        expect(task2.mock.calls.length).toBe(1);

        nowMock.mockReturnValue(5000);
        ticker.mock.calls[0][0]();

        expect(task1.mock.calls.length).toBe(3);
        expect(task2.mock.calls.length).toBe(2);

        removeTask1();

        nowMock.mockReturnValue(7000);

        ticker.mock.calls[0][0]();

        expect(task1.mock.calls.length).toBe(3);
        expect(task2.mock.calls.length).toBe(3);
    });

    it('Should NOT run tasks which are added in completed task', () => {
        const ticker = jest.fn();
        const taskManager = new TasksManager(ticker);

        const taskChild = jest.fn();
        const taskParent = jest.fn(() => taskManager.addInterval(taskChild, 500));

        const removeTaskParent = taskManager.addInterval(taskParent, 500);

        nowMock.mockReturnValue(500);
        ticker.mock.calls[0][0]();

        expect(taskParent.mock.calls.length).toBe(1);
        expect(taskChild.mock.calls.length).toBe(0);

        removeTaskParent();

        nowMock.mockReturnValue(1000);
        ticker.mock.calls[0][0]();

        expect(taskParent.mock.calls.length).toBe(1);
        expect(taskChild.mock.calls.length).toBe(1);
    });
});
