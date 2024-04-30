#!/usr/bin/python

import socket
import numpy as np
import time
import cv2

UDP_PORT = 999
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("192.168.137.1", UDP_PORT))

s=""
incoming = 8
pktSize = 55296
frameSize = 331776

numFrames = 8

while True:
  data, addr = sock.recvfrom(pktSize)
  s+= data

  print('Incoming data : ',incoming,'Addr  : ',addr)
  incoming+=1

  if incoming == numFrames:

    frame = np.fromstring (s, dtype=np.uint8)
    frame = frame.reshape(288,384,3)

    pts = np.array([[100,50],[100,80],[150,20],[200,200]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(frame,[pts],True,(0,255,255))

    cv2.imshow("server",frame)
    s=""
    
    name = "frame1.jpg"
    cv2.imwrite(name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
