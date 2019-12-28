import socket
import datetime

def server(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', port))
    print('Listening as {}'.format(sock.getsockname()))

    while True:
        data, address = sock.recvfrom(65535)
        text = data.decode('utf-8')
        print('The client at {} says\r\n{}'.format(address, text))
        data = text.encode('utf-8')
        with open('log.txt', 'a+', encoding='utf-8') as f:
            text = 'The time is {}'.format(datetime.time())
            f.write(str(text+'\n'+str(data)))
            f.write('\r\n\r\n')

        sock.sendto(data, address)

def client(host='117.136.57.245', port=8082, send=''):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data = send.encode('utf-8')
    s.sendto(data, (host, port))
    print('发送完成')
    data, addr = s.recvfrom(65535)
    print('{}'.format(data.decode('utf-8')))
    print('接受完成，关闭连接')
    s.close()
    
def server1(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((host, port))
    print('Server start')

    while 1:
        data, addr = s.recvfrom(2048)
        print('{} said:{}'.format(addr, data.decode('utf-8')))

server(8082)
