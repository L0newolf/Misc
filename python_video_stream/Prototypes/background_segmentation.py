import cv2
import detect_contours
import numpy as np
import imutils
import math
import matplotlib.pyplot as plt
from imutils.object_detection import non_max_suppression
from Prototypes.Utils import image

video_perc = 100

def backgroundSubtract(vfile):
    cap = cv2.VideoCapture(vfile)
    out = cv2.VideoWriter('motion_detected.avi', -1, 20.0, (384, 288))
    background_obj = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=50, detectShadows=0)
    frameCount = 0
    activity = 0
    while(True):
        success, frame = cap.read()
        if not success:
            break

        # frame = cv2.GaussianBlur(frame, (21, 21), 0)
        foreground = background_obj.apply(frame)
        cv2.imshow('segmented', foreground)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, element)
        foreground= cv2.dilate(foreground, None, iterations=2)


        img_, cnts, heirarchy = cv2.findContours(foreground.copy(), cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        for c in cnts:
            # if the contour is too small, ignore it
            # if cv2.contourArea(c) < frame.size * 1 / 1000:
            #     continue
            # (x, y, w, h) = cv2.boundingRect(c)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            rects.append(cv2.boundingRect(c))

        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.05)

        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

        out.write(frame)
        cv2.imshow('original', frame)

        activity += (cv2.countNonZero(foreground)) / foreground.size
        frameCount += 1
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    return activity / frameCount


def motionDetector(vfile,mask=True):
    cap = cv2.VideoCapture(vfile)
    firstFrame = None
    frameCount = 0
    activity = 0
    total = int((cap.get(cv2.CAP_PROP_FRAME_COUNT) / 100) * video_perc)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(5)
    out = cv2.VideoWriter('motion_detected_4Types.avi', -1, fps, (frame_width*2, frame_height*2))
    print(total)
    timeStamps = []
    percent_activity = []
    mask_not_set = True
    mask_img = None
    for i in range(0, total):
        success, frame = cap.read()
        if not success:
            break

        if mask == True:
            if mask_not_set == True:
                m, pts = image.Utils.get_bed_mask_from_roi(frame, "bed_frame")
                cv2.imwrite("mask.jpg" , m)
                mask_img = cv2.imread("mask.jpg")
                mask_not_set = False
            t = cv2.bitwise_and(mask_img, frame)
        else:
            t = frame

        gray = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (3,3), 0)
        canny = cv2.Canny(frame, 100, 255)
        if firstFrame is None:
            firstFrame = gray
            continue
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        # thresh = cv2.adaptiveThreshold(frameDelta, 25, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, element)
        thresh = cv2.dilate(thresh, None, iterations=3)
        img_, cnts, heirarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < frame.size * 1/10000:
                continue
            rects.append(cv2.boundingRect(c))
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            # (x, y, w, h) = cv2.boundingRect(c)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.05)

        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

        numpy_h1 = np.hstack((frame, cv2.cvtColor(frameDelta, cv2.COLOR_GRAY2BGR)))
        numpy_h2 = np.hstack((cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)))

        res = np.vstack((numpy_h1,numpy_h2))

        cv2.imshow('', res)
        out.write(res)
        activity += (cv2.countNonZero(thresh)) / thresh.size
        percent_activity.append((cv2.countNonZero(thresh)) / thresh.size)
        timeStamps.append(i)

        frameCount += 1
        firstFrame = gray
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    if frameCount == 0:
        frameCount = 1
    plt.plot(timeStamps, percent_activity)
    plt.show()
    return activity / frameCount

def peopleDetection(vFile):
    cap = cv2.VideoCapture(vFile)
    hog = cv2.HOGDescriptor()
    out = cv2.VideoWriter('pedestrians_detected.avi', -1, 20.0, (384, 288))
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = imutils.resize(frame,width = min(400,frame.shape[1]))
        orig = frame.copy()
        rects, weights = hog.detectMultiScale(frame, winStride = (4,4), padding = (8,8), scale = 1.05)

        for x,y,w,h in rects:
            cv2.rectangle(orig, (x,y), (x+w, y+h), (0, 0, 255), 2)

        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

        cv2.imshow("before nms ", orig)
        cv2.imshow("after nms" , frame)
        out.write(frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def rectify_video(vFile):
    cap = cv2.VideoCapture(vFile)
    prev_peak = None
    curr_peak = prev_peak
    while True:
        success, frame = cap.read()
        if not success:
            break
        if curr_peak < prev_peak:
            scale_contrast = (int)(prev_peak/curr_peak)
        else:
            scale_contrast = (int)(curr_peak/prev_peak)
        print(prev_peak, curr_peak, scale_contrast)
        frame = frame * scale_contrast
        prev_peak = max(prev_peak,curr_peak * scale_contrast)

        cv2.imshow("Original", orig)
        cv2.imshow("Rectified", frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # backgroundSubtract("D:/Datasets/new/051323.avi")
    backgroundSubtract("D:\Conex\PreSage\Prototypes\input\\193443.avi")
    # peopleDetection("D:/Datasets/new/051323.avi")
    # rectify_video("sample.avi")

