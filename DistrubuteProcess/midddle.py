from __future__ import absolute_import, unicode_literals
from multiprocessing import Queue, freeze_support
import socket, pickle
from multiprocessing import Process
import time
from DistrubuteProcess.Manager import QueueManager, SERVER

#任务队列
queue = Queue()
returnQueue = Queue()

# config
Distributed_server_addr = '192.168.0.103'
Distributed_server_port = 8001
authkey = 'zhangze'.encode('utf-8')

server_addr = '192.168.0.103'
port = 8000

def win_run(queue, returnQueue):
    # 将管理器注册，并进行任务绑定

    manager = QueueManager(address=(Distributed_server_addr, Distributed_server_port),
                           authkey=authkey)
    manager.server()

    task = manager.get_tasks_queue()
    result = manager.get_results_queue()
    returnResult = []

    while (1):
        returnResult = []
        try:
            tasks = queue.get()
            print('Get tasks :', tasks)
            for t in tasks:
                task.put(t)
            for i in range(len(tasks)):
                returnResult.append(result.get())

            returnQueue.put(returnResult)
        except:
            print('error, reset.')
            manager.shutdown()
            manager.start()
            task = manager.get_task_queue()
            result = manager.get_result_queue()


def GETserver(que, returnQueue):
    server = SERVER(server_addr, port)
    server.start()
    server.recv(que, returnQueue)


def main():
    # window下多进程可能有问题，添加这句话缓解
    freeze_support()
    print('start')
    p1 = Process(target=GETserver, args=(queue, returnQueue,))
    p2 = Process(target=win_run, args=(queue, returnQueue,))
    p1.start()
    p2.start()

if __name__ == '__main__':

    main()

