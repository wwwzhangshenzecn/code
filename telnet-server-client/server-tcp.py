import socket

# def recall(sock, lenght):
#     data =  b''
#     while len(data)<lenght:
#         more = sock.recv(lenght - len(data))
#         if not more:
#             raise EOFError('aa')
#
#         data += more
#     return data
#
# def Server(interface, port):
#     sock = socket.socket()
#
#     sock.bind((interface, port))
#     sock.listen(1)
#     print('Listening at {}'.format(sock.getsockname))
#     while True:
#         sc, sockname = sock.accept()
#         print('we accepted a connecton from', sockname)
#         print('Socket name: ',sc.getsockname)
#         print('Socket peer: ', sc.getpeername)
#         message = recall(sc,  16)
#         print('Incoming sixteen-octet message:', message)
#         sc.sendall(b'I have receive..')
#         sc.close()
#         print('Reply sent, socket closed.')
#
# if __name__=='__main__':
#     Server('', 8081)


from socketserver import BaseRequestHandler,TCPServer

class EchoHandler(BaseRequestHandler):
    def handle(self):
        print('Cient add:',self.client_address)

        while 1:
            msg = self.request.recv(8192)
            print('MSG: ',msg.decode('utf-8'))
            self.request.send('Yes!'.encode('utf-8'))

if __name__ == '__main__':
    serv = TCPServer(('',12345), EchoHandler)
    serv.serve_forever()