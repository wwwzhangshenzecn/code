from multiprocessing.managers import BaseManager
from multiprocessing import Queue, freeze_support
import socket,pickle
from multiprocessing import Pool,Process
from threading import Thread
queue =Queue()

#config
task_nums = 20
Distributed_server_addr = '127.0.0.1'
Distributed_server_port = 8001
authkey = 'zhangze'.encode('utf-8')

server_addr = '127.0.0.1'
port = 8000

# 收发队列
task_que = Queue(task_nums)
result_que = Queue(task_nums)


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

    def recv(self,queue):
        print('Get data.')
        while(1):
            client, addr  =self.sock.accept()
            data = self.recv_basic(client)
            data = pickle.loads(data)
            client.close()
            print('put : {}'.format(data))
            queue.put(data)


    def recv_basic(self,client):
        total_data = b''
        while True:
            data = client.recv(2048)
            if not data:break
            total_data+=data
        return total_data



def win_run(queue):
    # 将管理器注册，并进行任务绑定
    QueueManager.register('get_task_queue', callable=get_task)
    QueueManager.register('get_result_queue', callable=get_result)

    manager = QueueManager(address=(Distributed_server_addr, Distributed_server_port),
                           authkey=authkey)
    manager.start()
    task = manager.get_task_queue()
    result = manager.get_result_queue()

    while(1):
        try:
            tasks = queue.get()
            print('Manager get: ',tasks)
            for t in tasks:
                task.put(t)
            print('try get result')
            for i in range(20):
                print('reuslt is :', result.get(timeout=10))
        except:
            print('error, reset.')
            manager.shutdown()
            manager.start()
            task = manager.get_task_queue()
            result = manager.get_result_queue()

def GETserver(que):
    server = SERVER(server_addr, port)
    server.start()
    server.recv(que)


if __name__ == '__main__':
    # window下多进程可能有问题，添加这句话缓解
    freeze_support()
    print('start')
    p1 = Process(target=GETserver, args=(queue,))
    p2= Process(target=win_run, args=(queue, ))
    p1.start()
    p2.start()
