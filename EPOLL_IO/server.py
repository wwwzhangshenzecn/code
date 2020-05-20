import socket, select, collections,pickle

def recv_all(client, length):
    data = b''
    while len(data) < length:
        print(data)
        msg = client.recv(2048)
        if len(msg) == 0:break
        data += msg
    return data

def server(addr='', port=7999):

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((addr, port))
    server_sock.listen(20)

    print('server {}:{} start'.format(addr, port))

    epoll = select.epoll()
    epoll.register(server_sock.fileno(), select.EPOLLIN)

    fd_to_socket = {server_sock.fileno():server_sock,}
    socket_to_addr = {server_sock:'',}

    while 1:
        events = epoll.poll(timeout=10)
        if not events:
            continue

        for fd, event in events:
            sock = fd_to_socket[fd]
            if sock == server_sock:
                client, addr = sock.accept()
                print("{}:{} connection ...".format(client ,addr))
                epoll.register(client.fileno(), select.EPOLLIN)
                fd_to_socket[client.fileno()]=client
                socket_to_addr[client] = addr
            elif event & select.EPOLLIN:
                data_length = pickle.loads(sock.recv(2048))
                data = pickle.loads(recv_all(sock, data_length))
                print('{} : {}'.format(socket_to_addr[sock], data))
                epoll.modify(fd, select.EPOLLOUT)
            elif event & select.EPOLLOUT:
                data = pickle.dumps("server get it")
                sock.sendall(pickle.dumps(len(data)))
                sock.sendall(data)

                epoll.modify(fd, select.EPOLLIN)
            elif event & select.EPOLLHUP:
                print("{} is closed.".format(socket_to_addr[sock]))
                epoll.unregister(fd)
                fd_to_socket.pop(fd)
                socket_to_addr.pop(sock)
    epoll.unregister(server_sock.fileno())
    epoll.close()
    server_sock.close()

server()