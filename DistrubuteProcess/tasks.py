import time
from multipleworks.worker import Worker


def add(a, b=-1):
    print("woker: ", a, b)
    time.sleep(a)
    return a+b

def mul(a, b):
    return a*b

class test:
    def run(self):
        worker = Worker()
        worker.register([mul, add])
        worker.work()

if __name__ == '__main__':
    test().run()