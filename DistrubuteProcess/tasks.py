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
