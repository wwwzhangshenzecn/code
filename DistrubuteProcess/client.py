from multipleworks import client

if __name__ == '__main__':
    c = client.Client()
    c.connection()
    data = [['add',(1,2,3),{}] for _ in range(3)]

    result = c.run(data)
    print(result)