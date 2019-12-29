import socket, pickle,time

server_addr = '127.0.0.1'
port = 8000


def recv_basic(client, lenght):
    total_data = b''
    while len(total_data) < lenght:
        data = client.recv(2048)
        if not data: break
        total_data += data
    return total_data

def client(data):

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_addr, port))
    time.sleep(0.5) # 有一个小问题，如果不休眠，服务器可能会被卡在第二次接收状态
    data = pickle.dumps(data)
    sock.sendall(pickle.dumps(len(data)))
    sock.sendall(data)

    length = pickle.loads(sock.recv(2048*16))
    data = recv_basic(sock,length)
    data = pickle.loads(data)
    print('result :\n',data)
    sock.close()

if __name__ == '__main__':

    data = [ [1,{'a':1,'b':2}] for i in range(int(input()))]
    client(data)

