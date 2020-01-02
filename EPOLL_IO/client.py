import socket,pickle

def recv_all(client, length):
    data = b''
    while len(data) < length:
        msg = client.recv(2048)
        if len(msg) == 0:break
        data += msg
    return data
import time
def client(addr ,port):
    c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    c.connect((addr, port))
    time.sleep(0.5)
    while 1:
        data = input("input : ")
        data_pkl = pickle.dumps(data)
        c.sendall(pickle.dumps(len(data_pkl)))
        c.sendall(data_pkl)

        length = pickle.loads(c.recv(2048))
        data_recv = recv_all(c, length)
        msg = pickle.loads(data_recv)

        print(msg)

client('192.168.230.128',7999)
