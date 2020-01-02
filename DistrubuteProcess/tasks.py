from multipleworks import worker
def add(x, y, c):
    return x+y+c

if __name__ == '__main__':
    work =  worker.Worker()
    work.register(add)
    work.work()