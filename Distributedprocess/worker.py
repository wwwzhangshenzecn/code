from multiprocessing.managers import BaseManager
from multiprocessing import freeze_support, Queue
import time, random

class QueueManger(BaseManager):
    pass

QueueManger.register('get_task_queue')
QueueManger.register('get_result_queue')

server_addr = '192.168.0.103'
port = 8001
authkey = 'zhangze'.encode('utf-8')
m=QueueManger(address=(server_addr,port), authkey=authkey)
m.connect()

task = m.get_task_queue()
result = m.get_result_queue()

def work():
    print('work begin')
    while(1):
        try:
            t = task.get(True, timeout=5)
            print('Get task: ', t)
            time.sleep(random.randint(0,5))
            result.put(t)
        except:
            print('Empty task')
            # time.sleep(0.1)
            m.connect()
            task = m.get_task_queue()
            result = m.get_result_queue()

    print('begin')

if __name__ == '__main__':
    while 1:
        work()
