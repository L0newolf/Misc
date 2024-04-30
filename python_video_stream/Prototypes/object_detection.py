# import the necessary packages
import numpy as np,sys
import argparse
import glob
import cv2
from imutils.object_detection import non_max_suppression

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def gaussian_pyramid(image):
    G = image.copy()
    gpA = [G]
    for i in range(4):
        G = cv2.pyrDown(G)
        gpA.append(G)
    return gpA

def laplacian_pyramid(gpA):
    lpA = [gpA[3]]
    for i in range(3, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        L = cv2.subtract(gpA[i - 1], GE)
        lpA.append(L)
        cv2.imshow("Laplacian " + str(i), L)
    return lpA

def hog(image):
    im = np.float32(image)/255
    gx = cv2.Sobel(im,cv2.CV_32F,1,0,ksize=1)
    gy = cv2.Sobel(im,cv2.CV_32F,0,1,ksize=1)
    mag, angle = cv2.cartToPolar(gx,gy,angleInDegrees=True)
    cv2.imshow("HOG Magnitude", mag * 10)
    return mag

def K_Means(img,K=2):
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    res2 = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
    return res2

def detect_collision(a, b):
    x1,y1,w1,h1 = a
    x2,y2,w2,h2 = b
    return (abs(x1 - x2) * 2 < (w1 + w2)) and (abs(y1 - y2) * 2 < (h1 + h2))

def get_iou(a,b):
    x1, y1, w1, h1 = a
    x3, y3, w3, h3 = b

    x2 = x1 + w1
    y2 = y1 + h1
    x4 = x3 + w3
    y4 = y3 + h3

    x_left = max(x1,x3)
    y_top = max(y1,y3)
    x_right = min(x2,x4)
    y_bottom = min(y2,y4)

    # No intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right-x_left) * (y_bottom-y_top)

    a_area = w1*h1
    b_area = w3 * h3

    iou = intersection_area/float(a_area + b_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0

    return iou


def get_moments(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = cv2.GaussianBlur(gray, (15, 15), 0)
    ret, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edged = auto_canny(gray)
    img_, contours, heirarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    tooSmall = gray.size * 5 / 1000
    tooBig = gray.size * 100 / 100
    moments = []
    centroids = []
    ellipsoids = []
    for c in contours:
        rects.append(cv2.boundingRect(c))
        M = cv2.moments(c)
        moments.append(M)
        m1 = 0
        m2 = 0
        # Calculating bary-center of object using 1st central moments along x & y axes
        if M["m00"] != 0:
            m1 = int(M['m10'] / M['m00'])
            m2 = int(M['m01'] / M['m00'])

        centroids.append([m1,m2])
        cv2.circle(image, (m1, m2), 2, (0, 0, 255), -1)

        # Calculating object orientation from 2nd central moments
        # theta = 0.5 * np.arctan ((2 * M['mu11'])/ (M['mu20'] - M['mu02']) )
        # theta = theta/np.pi * 180
        if len(c) > 5:
            ellipse = cv2.fitEllipse(c)
            cv2.ellipse(image, ellipse, (0, 250, 0), 2)
        else:
            ellipse = None

        ellipsoids.append(ellipse)
    return contours, moments, centroids, ellipsoids


## this function will update the input image with the bounding boxes of the objects found overlayed on top of the image
def detect_object(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (15, 15), 0)
    ret, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edged = auto_canny(gray)
    img_, contours, heirarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("Canny ", edged)
    rects = []
    tooSmall = gray.size * 5/ 1000
    tooBig = gray.size * 100 / 100
    for c in contours:
        rects.append(cv2.boundingRect(c))

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.01)
    results = []
    for (xA, yA, xB, yB) in pick:
        if (abs(xB - xA) * abs(yB - yA)) < tooSmall:
            continue
        results.append((xA,yA,xB-xA,yB-yA))
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 0, 255), 2)


    return gray,results

def merge_rect(a,b):
    x1, y1, w1, h1 = a
    x2, y2, w2, h2 = b
    left = min(x1,x2)
    top = min(y1,y2)
    right = max(x1 + w1, x2 + w2)
    bottom = max(y1 + h1, y2 + h2)
    return (left, top, right - left, bottom - top)

def track_objects(video):
    cap = cv2.VideoCapture(video)
    first = True
    myObjTrackers = []
    myObjects = []
    init_bbox = (287, 23, 86, 320)
    while True:
        success, image = cap.read()
        if not success:
            break
        res, rects = detect_object(image.copy())
        for rect in rects:
            newObject = True
            obj_index = -1
            for obj in myObjects:
                if get_iou(obj,rect)>0.8:
                    newObject = False
                    obj_index = myObjects.index(obj)
                    break
            if newObject == True:
                tracker = createTracker("KCF")
                init_bbox = rect
                # init_bbox = cv2.selectROI(image, False)
                # print(rects[0], init_bbox)
                tracker.init(image,init_bbox)
                myObjTrackers.append(tracker)
                myObjects.append(rect)
            else:
                ok, bbox = myObjTrackers[obj_index].update(image)
                if ok:
                    # Tracking success
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(image, p1, p2, (255, 0, 0), 2, 1)
                    myObjects[obj_index] = bbox#merge_rect(rect,bbox)
                    x,y,w,h = myObjects[obj_index]
                    # cv2.rectangle(image, (int(x),int(y)) , (int(x)+int(w), int(y)+int(h)), (0, 255, 0), 2, 1)
                    (x,y) = p2
                    p2 = (x-15,y-5)
                    cv2.putText(image, str(obj_index), p2, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))

        cv2.imshow("detecting ", res)
        cv2.imshow("tracking ",image)

        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def createTracker(tracker_type="MIL"):
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
    return tracker


def detect_objects(video):
    cap = cv2.VideoCapture(video)
    total = int((cap.get(cv2.CAP_PROP_FRAME_COUNT) / 100) * 100)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(5)
    out = cv2.VideoWriter('objects_detected_2Types.avi', -1, fps, (frame_width*2 , frame_height ))
    print(total)
    for i in range(0, total):
        success, image = cap.read()
        if not success:
            break
        gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)
        ret, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        edged = auto_canny(gray)
        res, rects = detect_object(image)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        cv2.imshow("Objects detected ",image)
        numpy_h1 = np.hstack((image, cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)))
        out.write(numpy_h1)
    out.release()
    cap.release()
    cv2.destroyAllWindows()

def display_moments(video):
    cap = cv2.VideoCapture(video)
    total = int((cap.get(cv2.CAP_PROP_FRAME_COUNT) / 100) * 100)
    for i in range(0, total):
        success, image = cap.read()
        if not success:
            break
        mcontours, moments, centroids, ellipsoids = get_moments(image)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        cv2.imshow(" Moments ",image)
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    display_moments("D:/Datasets/TTSH_Videos/Video_Patient_003_Bed_03_07-06-1655-10-06-1300/17/06/07/193443.avi")
