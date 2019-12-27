from multiprocessing.managers import BaseManager
from multiprocessing import Queue, freeze_support
import socket, pickle
from multiprocessing import Pool, Process
from threading import Thread
import time

# 服务器任务队列
queue = Queue()
returnQueue = Queue()


# config
task_nums = 20
Distributed_server_addr = '192.168.0.103'
Distributed_server_port = 8001
authkey = 'zhangze'.encode('utf-8')

server_addr = '192.168.0.103'
port = 8000

# 分布式管理收发队列
task_que = Queue()
result_que = Queue()


def get_task():
    return task_que


def get_result():
    return result_que


class QueueManager(BaseManager):
    pass


class SERVER:
    def __init__(self, addr, port):
        self.addr = addr
        self.port = port
        self.sock = None

    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(('', self.port))
        self.sock.listen(5)

    def recv(self, queue, returnQueue):
        print('Get data.')
        while (1):
            client, addr = self.sock.accept()
            print('\n\n\n\nuser {} connection .'.format(addr))
            start = time.time()
            lenght = pickle.loads(client.recv(1024))

            data = self.recv_basic(client, lenght)

            data = pickle.loads(data)
            print('tasks len : {}'.format(len(data)))
            queue.put(data)

            data = returnQueue.get()
            data = pickle.dumps(data)
            client.sendall(pickle.dumps(len(data)))

            client.sendall(data)
            end = time.time()
            print('user {} close. tieme : {} s'.format(addr, end-start))
            client.close()

    def recv_basic(self, client, lenght):
        total_data = b''
        while len(total_data)<lenght:
            data = client.recv(2048*16)
            if not data: break
            total_data += data
        return total_data


def win_run(queue,returnQueue):
    # 将管理器注册，并进行任务绑定
    QueueManager.register('get_task_queue', callable=get_task)
    QueueManager.register('get_result_queue', callable=get_result)

    manager = QueueManager(address=(Distributed_server_addr, Distributed_server_port),
                           authkey=authkey)
    manager.start()
    task = manager.get_task_queue()
    result = manager.get_result_queue()
    returnResult = []
    while (1):
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


def GETserver(que,returnQueue):
    server = SERVER(server_addr, port)
    server.start()
    server.recv(que, returnQueue)


if __name__ == '__main__':
    # window下多进程可能有问题，添加这句话缓解
    freeze_support()
    print('start')
    p1 = Process(target=GETserver, args=(queue,returnQueue,))
    p2 = Process(target=win_run, args=(queue,returnQueue,))
    p1.start()
    p2.start()
