#!/usr/bin/env python

import random
import socket, select
from time import gmtime, strftime
from random import randint
import time

SERVER = '192.168.137.10'
PORT = 8899

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (SERVER, PORT)
sock.connect(server_address)

sock.close()