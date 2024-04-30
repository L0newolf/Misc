#!/usr/bin/env python

import random
import socket, select
from time import gmtime, strftime
from random import randint

imgcounter = 0
basename = "image%s.png"

HOST = '192.168.137.1'
PORT = 6666

connected_clients_sockets = []

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(10)

connected_clients_sockets.append(server_socket)

print("Here 1")

while True:

    print("Here 2")

    print("Waiting for connection")
    sockfd, client_address = server_socket.accept()
    print("Got connection from ",client_address)
    connected_clients_sockets.append(sockfd)
    imgcounter += 1

    try:
        print("Getting")
        myfile = open(basename % imgcounter, 'wb')
        myfile.write(data)
        data = sockfd.recv(40960000)
        if not data:
            myfile.close()
            break
        myfile.write(data)
        myfile.close()
        sockfd.shutdown()
        print("Got image")

        """
        if data:
            print("Here 3")
            if data.startswith('SIZE'):
                print("Here 4")
                tmp = txt.split()
                size = int(tmp[1])

                print('got size',size)
                sockfd.sendall("GOT SIZE")

            elif data.startswith('BYE'):
                print("Here 4")
                sockfd.shutdown()
            else :
                print("Here 5")
                myfile = open(basename % imgcounter, 'wb')
                myfile.write(data)
                data = sockfd.recv(40960000)
                if not data:
                    myfile.close()
                    break
                myfile.write(data)
                myfile.close()
                sockfd.sendall("GOT IMAGE")
                sockfd.shutdown()
            """    
    except:
        sockfd.close()
        connected_clients_sockets.remove(sockfd)
        continue
    print("Here 6")    
    
"""
    read_sockets, write_sockets, error_sockets = select.select(connected_clients_sockets, [], [])

    print("Here 3")

    for sock in read_sockets:

        print("Here 4")

        if sock == server_socket:

            print("Waiting for connection")
            sockfd, client_address = server_socket.accept()
            print("Got connection from %s" % client_address)
            connected_clients_sockets.append(sockfd)

        else:
            try:

                data = sock.recv(4096)
                txt = str(data)

                if data:

                    if data.startswith('SIZE'):
                        tmp = txt.split()
                        size = int(tmp[1])

                        print('got size')

                        sock.sendall("GOT SIZE")

                    elif data.startswith('BYE'):
                        sock.shutdown()

                    else :

                        myfile = open(basename % imgcounter, 'wb')
                        myfile.write(data)

                        data = sock.recv(40960000)
                        if not data:
                            myfile.close()
                            break
                        myfile.write(data)
                        myfile.close()

                        sock.sendall("GOT IMAGE")
                        sock.shutdown()
            except:
                sock.close()
                connected_clients_sockets.remove(sock)
                continue
        imgcounter += 1
"""
server_socket.close()