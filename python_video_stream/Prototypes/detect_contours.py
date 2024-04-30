# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 14:08:30 2017

@author: div_1
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from imutils.object_detection import non_max_suppression

def get_blobs(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img= cv2.GaussianBlur(img, (21, 21), 0)
    img = threshold(img)
    img_, contours, heirarchy = cv2.findContours(img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    significant = []
    tooSmall = img.size * 1/1000
    tooBig   = img.size * 100/100
    for c in contours:
        area = cv2.contourArea(c)
        if area <= tooBig  and area > tooSmall:
            significant.append(c)
    return significant


    # level1 = []
    # for i, tupl in enumerate(heirarchy[0]):
    #    # Each array is in format (Next, Prev, First child, Parent)
    #    if tupl[3] == -1:
    #        tupl = np.insert(tupl, 0, [i])
    #        level1.append(tupl)
    #    significant = []
    #    tooSmall = img.size * 5 / 100
    #    for tupl in level1:
    #        contour = contours[tupl[0]];
    #        area = cv2.contourArea(contour)
    #        if area > tooSmall:
    #            significant.append([contour, area])
    #    significant.sort(key=lambda x: x[1])
    #    print ([x[1] for x in significant]);
    # return [x[0] for x in significant];

def laplacian(img):
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    if __name__ == "__main__":
        dispImg("laplacian", laplacian)
    return laplacian

def edgedetect (channel):
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    sobel = np.hypot(sobelX, sobelY)

    sobel[sobel > 255] = 255
    return sobel
    
def get_brightestSpot(img, radius):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (radius, radius), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    return maxLoc

def plot_histogram(img,name = ""):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.figure(name)
    plt.subplot(121), plt.imshow(img,"gray")
    plt.subplot(122), plt.plot(hist)
    plt.show()

def get_histogram_peak(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    return np.argmax(hist)


def equalize_histogram(img,name = ""):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    equ_hist = cv2.calcHist([equ], [0], None, [256], [0, 256])
    plt.figure(name)
    plt.subplot(121), plt.imshow(equ, 'gray')
    plt.subplot(122), plt.plot(equ_hist)

def normalize_histogram(img,name = ""):
    norm = img.copy()
    cv2.normalize(img, dst=norm, alpha=50, beta=150, norm_type=cv2.NORM_MINMAX)
    norm_hist = cv2.calcHist([norm], [0], None, [256], [0, 256])
    plt.figure(name)
    plt.subplot(121), plt.imshow(norm, 'gray')
    plt.subplot(122), plt.plot(norm_hist)

def dispImg(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)


def threshold(img):
    img = cv2.medianBlur(img,7)

    thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    if __name__ == "__main__":
        dispImg("binary threshold", thresh)

    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    th3 = cv2.morphologyEx(th3,cv2.MORPH_OPEN, element)
    th3 = cv2.bitwise_not(th3)
#     th3 = cv2.morphologyEx(th3,cv2.MORPH_OPEN, element)
    if __name__ == "__main__":
        dispImg("adaptive threshold",th3)
    return th3
    
def detectContours(img):
    contours = get_blobs(img)
    rects = []
    for c in contours:
        rects.append(cv2.boundingRect(c))

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.15)

    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(img, (xA, yA), (xB, yB), (0, 0, 255), 2)
    # cv2.drawContours(img,contours, -1, (0,255,0), 3)

    if __name__ == "__main__":
        dispImg("contours", img)
    return img

if __name__ == "__main__":
    img = cv2.imread("white_after.jpg")
    img2 = cv2.imread("before.jpg")
    p1 = get_histogram_peak(img)
    p2 = get_histogram_peak(img2)
    cv2.imshow("Original Balck", img)
    print((int)(p2/p1)/2)
    img = img * 3 #(int)(p2/(1.5*p1))
    cv2.imshow("1",img)
    cv2.imshow("2",img2)

    detectContours(img)
