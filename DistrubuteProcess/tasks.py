import time

def work(a, b=-1):
    print("woker: ", a, b)
    time.sleep(a)
    return a+b