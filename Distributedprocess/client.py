import socket, pickle

server_addr = '127.0.0.1'
port = 8000

if __name__ == '__main__':
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_addr, port))
    data = [i for i in range(20)]
    data = pickle.dumps(data)

    sock.sendall(data)
    sock.close()