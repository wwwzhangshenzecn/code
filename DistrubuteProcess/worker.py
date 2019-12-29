'''
任务参数 必须以列表嵌套元组和字典的形式给出
例子：
import time
from DistrubuteProcess.worker import Worker


def add(a, b=-1):
    print("woker: ", a, b)
    time.sleep(a)
    return a+b

def mul(a, b):
    return a*b

worker  = Worker()
worker.register(mul)
worker.work()

'''

from DistrubuteProcess.Manager import QueueManager


class Worker:

    def __init__(self):
        self.func = {}
        self.server_addr = '192.168.0.103'
        self.port = 8001
        self.authkey = 'zhangze'.encode('utf-8')
        self.m = QueueManager(address=(self.server_addr, self.port), authkey=self.authkey)
        self.m.client()
        self.task = self.m.get_tasks_queue()
        self.result = self.m.get_results_queue()

    def register(self, func):
        if isinstance(func, list):
            for f in func:
                self.func[f.__name__] = f
        else:
            self.func[func.__name__] = func

    def work(self):
        funcname_error = '不存在此函数 ：{}'
        print('work begin')
        while (1):
            try:
                t = self.task.get(True, timeout=5)
                funcname, args, kwargs = t[0], t[1], t[2]
                print('Get task: {}, {}, {}'.format(funcname, args, kwargs))
                if funcname not in self.func.keys():
                    self.result.put(funcname_error.format(funcname))
                else:
                    re = self.func[funcname](*args, **kwargs)
                    self.result.put(re)
            except:
                print('Empty tasks')
                self.m.client()
                self.task = self.m.get_tasks_queue()
                self.result = self.m.get_results_queue()

        print('work end')
