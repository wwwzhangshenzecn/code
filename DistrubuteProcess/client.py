import socket, pickle,time

def recv_basic(client, lenght):
    total_data = b''
    while len(total_data) < lenght:
        data = client.recv(2048)
        if not data: break
        total_data += data
    return total_data

class Client:

    def __init__(self):
        self.sock=None
    def connection(self,addr='127.0.0.1',port=8000):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((addr, port))
        time.sleep(0.5)

    def run(self, data=None):
        data = pickle.dumps(data)
        self.sock.sendall(pickle.dumps(len(data)))
        self.sock.sendall(data)

        length = pickle.loads(self.sock.recv(2048*16))
        data = recv_basic(self.sock,length)
        data = pickle.loads(data)
        self.sock.close()
        return data


class testClient:
    def run(self ,data=None):
        if data == None:
            data = [ ['mul',(i,i)] for i in range(int(input()))]+\
                [['add', (i, i)] for i in range(int(input()))]
        c = Client()
        c.connection()
        data = c.run(data)
        print(data)

if __name__ == '__main__':
    testClient().run()