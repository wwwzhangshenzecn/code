from multiprocessing.managers import BaseManager
from multiprocessing import freeze_support, Queue
import time, random

class QueueManger(BaseManager):
    pass

QueueManger.register('get_task_queue')
QueueManger.register('get_result_queue')

server_addr = '127.0.0.1'
m=QueueManger(address=(server_addr,8001), authkey='zhangze'.encode('utf-8'))
m.connect()

task = m.get_task_queue()
result = m.get_result_queue()

print('work begin')
while(1):
    try:
        image_url = task.get(True, timeout=5)
        print('Get task: ', image_url)
        time.sleep(random.randint(0,5))
        result.put('sucess------>>>>>{}'.format(image_url))
    except:
        print('Empty task')
        time.sleep(5)

print('work end')