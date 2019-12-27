import socket, pickle

server_addr = '127.0.0.1'
port = 8000


def recv_basic(client, lenght):
    total_data = b''
    while len(total_data) < lenght:
        data = client.recv(2048)
        if not data: break
        total_data += data
    return total_data

if __name__ == '__main__':
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_addr, port))
    n = input("输入任务数量: ")
    data = [i for i in range(int(n))]
    data = pickle.dumps(data)
    sock.sendall(pickle.dumps(len(data)))
    sock.sendall(data)

    length = pickle.loads(sock.recv(2048*16))
    data = recv_basic(sock,length)
    data = pickle.loads(data)
    sock.close()
