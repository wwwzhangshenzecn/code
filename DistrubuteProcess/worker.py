from DistrubuteProcess.Manager import QueueManager
import time, random
from DistrubuteProcess import tasks

server_addr = '192.168.0.103'
port = 8001

authkey = 'zhangze'.encode('utf-8')
m=QueueManager(address=(server_addr,port), authkey=authkey)
m.client()

task = m.get_tasks_queue()
result = m.get_results_queue()

def work():
    print('work begin')
    while(1):
        try:
            t = task.get(True, timeout=5)
            print('Get task: ', t)
            tasks.work(t)
            result.put(t)
        except:
            print('Empty tasks')
            m.connect()
            task = m.get_tasks_queue()
            result = m.get_results_queue()

    print('begin')

if __name__ == '__main__':
    while 1:
        work()
