#!/usr/bin/python

import socket
import numpy as np
import time
import cv2
from os import listdir
from os.path import isfile, join
import pickle
import sys

UDP_IP = "192.168.10.1"
UDP_PORT = 999
UDP_CLIENT = "192.168.10.10"
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

while(True):

    connCmd = 'Send file list'
    print connCmd

    if (sock.sendto (connCmd,(UDP_CLIENT, UDP_PORT))) == 0 :
        print 'Connection failed '
        break
    else :
        print 'Waiting for file list from client .. '
        data, addr = sock.recvfrom(1024)
        fileslist = pickle.loads(data)
        print fileslist
        break
