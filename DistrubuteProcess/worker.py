'''
任务参数 必须以列表嵌套元组和字典的形式给出
'''

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
            args, kwargs = task.get(True, timeout=5)
            print('Get task: {}, {}'.format(args, kwargs))
            re = tasks.work(*args, **kwargs)
            result.put(re)
        except:
            print('Empty tasks')
            m.connect()
            task = m.get_tasks_queue()
            result = m.get_results_queue()

    print('work end')

if __name__ == '__main__':
    work()
