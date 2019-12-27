from multiprocessing.managers import BaseManager
from multiprocessing import Queue, freeze_support

task_nums = 10

# 收发队列
task_que = Queue(task_nums)
result_que = Queue(task_nums)


def get_task():
    return task_que


def get_result():
    return result_que


class QueueManager(BaseManager):
    pass


def win_run():
    # 将管理器注册，并进行任务绑定
    QueueManager.register('get_task_queue', callable=get_task)
    QueueManager.register('get_result_queue', callable=get_result)

    manager = QueueManager(address=('127.0.0.1', 8001),
                           authkey='zhangze'.encode('utf-8'))

    manager.start()
    try:
        task = manager.get_task_queue()
        result = manager.get_result_queue()

        for url in ['ImageUrl_' + str(i) for i in range(20)]:
            task.put(url)

        print('try get result')
        for i in range(20):
            print('reuslt is :', result.get(timeout=10))
    except:
        manager.shutdown()

if __name__ == '__main__':
    # window下多进程可能有问题，添加这句话缓解
    freeze_support()
    win_run()
