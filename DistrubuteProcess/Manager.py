from __future__ import absolute_import, unicode_literals
from multiprocessing.managers import BaseManager
from multiprocessing import Queue
import pickle, socket, time
import functools


class QueueManager(BaseManager):

    def get_tasks(self):
        return self.tasks

    def get_results(self):
        return self.results

    def __init__(self,tasks_num=-1,results_num=-1,  *args, **kwargs):
        self.tasks_num = tasks_num
        self.results_num = results_num
        self.tasks = Queue(self.tasks_num)
        self.results= Queue(self.results_num)
        super(QueueManager, self).__init__(*args, **kwargs)
        super().register('tasks_queue', callable=self.get_tasks)
        super().register('results_queue', callable=self.get_results)


    def server(self):
        self.start()

    def client(self):
        self.connect()

    def get_tasks_queue(self):
        return self.tasks_queue()

    def get_results_queue(self):
        return self.results_queue()


class SERVER:
    def __init__(self, addr, port):
        self.addr = addr
        self.port = port
        self.sock = None

    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('', self.port))
        self.sock.listen(5)

    def recv(self, queue, returnQueue):
        print('Get data.')
        while (1):
            client, addr = self.sock.accept()
            try:
                print('\n\nuser {} connection .'.format(addr))
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
                print('user {} close. time : {} s'.format(addr, end - start))

            except:
                print('此任务异常...')
            finally:
                client.close()

    def recv_basic(self, client, lenght):
        total_data = b''
        while len(total_data) < lenght:
            data = client.recv(2048 * 2)
            total_data += data
            if not data: break
        return total_data



