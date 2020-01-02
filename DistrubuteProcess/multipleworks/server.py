from __future__ import absolute_import, unicode_literals
from multiprocessing import Queue, freeze_support
from multiprocessing import Process
from .Manager import QueueManager, SERVER

# 任务队列
queue = Queue()
returnQueue = Queue()

def win_run(queue, returnQueue, addr, port, authkey):
    # 将管理器注册，并进行任务绑定
    manager = QueueManager(address=(addr, port),
                           authkey=authkey.encode('utf-8'))
    manager.server()

    task = manager.get_tasks_queue()
    result = manager.get_results_queue()

    argserror = '\n任务格式必须为: [ name:str,args:tuple, kwargs:dict ]\n'

    while (1):
        returnResult = []
        try:
            tasks = queue.get()
            count = 0

            for t in tasks:
                if not isinstance(t, list) or len(t) > 3 or len(t) == 0:
                    print('抛弃任务： {}'.format(t))
                    returnResult.append(argserror)
                    continue

                elif not isinstance(t[0], str):
                    print('抛弃任务： {}'.format(t))
                    returnResult.append(argserror)
                    continue

                if len(t) == 2:
                    if isinstance(t[1], tuple):
                        t.append({})
                    if isinstance(t[1], dict):
                        t.insert(1, tuple())

                if not isinstance(t[1], tuple) or not isinstance(t[2], dict) or not isinstance(t[0], str):
                    print('抛弃任务： {}'.format(t))
                    returnResult.append(argserror)
                    continue

                count += 1
                print('ready tasks: ', t)
                task.put(t)

            for i in range(count):
                returnResult.append(result.get(timeout=10))

            returnQueue.put(returnResult)
        except:
            print('error, reset.')
            manager.shutdown()
            manager.start()
            task = manager.get_task_queue()
            result = manager.get_result_queue()


def GETserver(que, returnQueue, addr='', port=8000):
    server = SERVER(addr, port)
    server.start()
    server.recv(que, returnQueue)


class middle:
    def run(self, manager_addr='127.0.0.1', manager_port=8001,authkey='zhangze',
            server_add='', server_port=8000):
        # window下多进程可能有问题，添加这句话缓解
        freeze_support()
        print('start')
        p1 = Process(target=GETserver, args=(queue, returnQueue,server_add, server_port))
        p2 = Process(target=win_run, args=(queue, returnQueue,manager_addr,manager_port,authkey))
        p1.start()
        p2.start()

#
# if __name__ == '__main__':
#     middle().run()