#!/usr/bin/python

import socket
import numpy as np
import cv2
import time
import sys
              
UDP_PORT = 999        
UDP_IP = "192.168.137.1" 

cap = cv2.VideoCapture('bed_exit.avi')

outgoing = 1;

numPackets = 8

while(True):
	ret, frame = cap.read()
	cv2.imshow('client',frame)
	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	d = frame.flatten ()
	s = d.tostring ()
	frameSize = sys.getsizeof(s)
	pktSize = int(frameSize/numPackets)
	for i in range(numPackets):
		if (sock.sendto(s[i*pktSize:(i+1)*pktSize],(UDP_IP, UDP_PORT))) == 0 :
			print('Connection failed ')
			break
		else :
			print('Outgoing data : ',outgoing)
			outgoing+=1

		time.sleep(0.015)	

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	print('Frame sent')	
	time.sleep(0.16)
	
cap.release()
cv2.destroyAllWindows()