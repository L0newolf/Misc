#!/usr/bin/env python

import random
import socket, select
from time import gmtime, strftime
from random import randint
import time

image = "frame1.jpg"

HOST = '127.0.0.1'
PORT = 6666

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (HOST, PORT)
sock.connect(server_address)

try:

    # open image
    print("Sending image")
    myfile = open(image, 'rb')
    bytes = myfile.read()
    size = len(bytes)
    sock.sendall(bytes)
    time.sleep(1)
    print("Sent image")

    """
    # send image size to server
    print ('Sending image size = %s' % size)
    sock.sendall("SIZE %s" %size)
    time.sleep(1)
    answer = sock.recv(4096)

    print ('answer = %s' % answer)

    # send image to server
    if answer == 'GOT SIZE':
        sock.sendall(bytes)
        time.sleep(1)

        # check what server send
        answer = sock.recv(4096)
        print ('answer = %s' % answer)

        if answer == 'GOT IMAGE' :
            sock.sendall("BYE BYE ")
            print ('Image successfully send to server')
    """

    myfile.close()

finally:
    sock.close()