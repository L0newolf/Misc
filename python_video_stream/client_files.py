#!/usr/bin/python

import socket
import numpy as np
import time
import cv2
from os import listdir
from os.path import isfile, join
import pickle



mypath = "/home/pi/stream"

UDP_IP = "192.168.10.10"
UDP_PORT = 999
UDP_SERVER = "192.168.10.1"
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))


while True:
      print "Waiting for connection from server "
      data, addr = sock.recvfrom(1024)
      print "Got connection from server : ",addr[0]," with command : ",data
      if data == 'Send file list':
          onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
          print onlyfiles
          data_string = pickle.dumps(onlyfiles)
          sock.sendto (data_string,(UDP_SERVER, UDP_PORT))
          break
      else:
          print "Invalid command from server"
          break
