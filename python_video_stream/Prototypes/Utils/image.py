import cv2
import tensorflow as tf
import random
import os
import numpy as np

num = 0
class Utils(object):
    @staticmethod
    def extract_frame(video,frame_number):
        video.set(1,frame_number)
        ret,frame = video.read()
        return frame

    @staticmethod
    def crop_frame(frame,box):
        (x_tl,y_tl,x_br,y_br) = box
        cropped_frame = frame[y_tl:y_br,x_tl:x_br]
        return cropped_frame

    @staticmethod
    def extract_cropped_frame(video,frame_number,box):
        f = Utils.extract_frame(video,frame_number)
        Utils.crop_frame(f,box)

    @staticmethod
    def cv2_tf_encoded_image(frame):
        # a = random.randint(1, 1000001)
        # b = random.randint(1,100000)
        # num = (a + b) % 832040 #1335331
        path = "temp_encoded_%d.jpg"%num
        cv2.imwrite(path,frame)
        with tf.gfile.GFile(path, 'rb') as fid:
            encoded_image = fid.read()
        os.remove(path)
        return encoded_image

    @staticmethod
    def cv2_tf_decoded_image(frame):
        # num = random.randint(1, 1000001)
        path = "temp_decoded_%d.jpg"%num
        cv2.imwrite(path,frame)
        with tf.gfile.FastGFile(path, 'rb') as fid:
            decoded_image =  fid.read()
        # os.remove(path)
        return decoded_image

    @staticmethod
    def get_bed_mask_from_roi(frame, window_name="polygon"):
        pd = PolygonDrawer(window_name=window_name, frame=frame, max_points=4)
        mask, pts = pd.run()
        return mask, pts


class PolygonDrawer(object):
    def __init__(self, window_name,frame = None, max_points = -1 ):
        self.window_name = window_name # Name for our window
        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        self.final_line_color = (255, 255, 255)
        self.working_line_color = (127,127,127)
        if frame is not None:
            self.frame = frame
            height, width, channels = self.frame.shape
            self.canvas_size = (height, width)
        else:
            self.canvas_size = (600, 800)
            self.frame = np.ones(self.canvas_size, np.uint8)
        self.max_points = max_points #stop the polygon roi once you've hit max_points


    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done: # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
            if self.max_points != -1 and len(self.points) == self.max_points:
                self.points.append(self.points[0])
                self.done = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            print("Completing polygon with %d points." % len(self.points))
            self.done = True


    def run(self):
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self.window_name, self.frame)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while(not self.done):
            canvas = self.frame.copy()
            if (len(self.points) > 0):
                cv2.polylines(canvas, np.array([self.points]), False, self.final_line_color, 1)
                cv2.line(canvas, self.points[-1], self.current, self.working_line_color)
            cv2.imshow(self.window_name, canvas)
            if cv2.waitKey(50) == 27: # ESC hit
                self.done = True

        canvas = np.zeros(self.canvas_size, np.uint8)
        if (len(self.points) > 0):
            cv2.fillPoly(canvas, np.array([self.points]), self.final_line_color)
        cv2.destroyWindow(self.window_name)
        return canvas,self.points

