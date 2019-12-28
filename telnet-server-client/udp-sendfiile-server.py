import json
import os
import socket
import struct
i = 0
while 1:
    print('---------receive {} files: {}'.format(i ,i))
    ip_port = ('', 12345)
    sk = socket.socket()
    sk.bind(ip_port)
    sk.listen()

    buffer = 1024
    conn, addr = sk.accept()

    pack_len = conn.recv(4)
    head_len = struct.unpack('i', pack_len)[0]
    json_head = conn.recv(head_len).decode('utf-8')
    head = json.loads(json_head)
    filesize = head['filesize']
    with open(head['filename'], 'wb') as f:
        while filesize:
            print(filesize)
            if filesize >= buffer:
                content = conn.recv(buffer)
                filesize -= buffer
                f.write(content)
            else:
                content = conn.recv(filesize)
                f.write(content)
                break
    conn.close()
    sk.close()
    i+=1
