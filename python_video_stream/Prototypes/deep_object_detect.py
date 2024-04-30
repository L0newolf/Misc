############### NOTE:
############### Requires - Tensorflow + Tensorflow\models
############### This is to be added to $PATH_TO_TENSORFLOW$\models\research\object_detection
############### Assumes tensorflow\models\slim is added to env path
############### Simply uses the pre-trained object detection models available (SSD, SSD-MobileNet, Fast R-CNN, Inception) to display detected objects
###############


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# What model to download.
# MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_NAME = 'ssd_inception_v2_coco_11_06_2017' #very good!
# MODEL_NAME = 'rfcn_resnet101_coco_11_06_2017' #too slow
# MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017' #too slow
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
sys.path.append("..")
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

fname = PATH_TO_CKPT
if not os.path.isfile(fname) :
    print("Downloading frozen model")
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
      file_name = os.path.basename(file.name)
      if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())


print("loading graph")
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

allowed_ext = [".avi", ".mp4", ".wmv", ".flv", ".mov"]
PATH_TO_TEST_IMAGES_DIR = 'D:\Datasets\scenarios'
for subdir, dirs, files in os.walk(PATH_TO_TEST_IMAGES_DIR):
    for file in files:
        video_file = subdir + os.sep + file
        if video_file.endswith(tuple(allowed_ext)):
            # Size, in inches, of the output images.
            IMAGE_SIZE = (12, 8)
            filename,ext = video_file.split(".")
            print("running graph on video stream")

            with detection_graph.as_default():
              with tf.Session(graph=detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                writer = tf.summary.FileWriter("log", sess.graph)

                cap = cv2.VideoCapture(video_file)
                frame_width = int(cap.get(3))
                frame_height = int(cap.get(4))
                vout = cv2.VideoWriter(filename + "_out." + ext, -1, 8, (frame_width + 20, frame_height + 30))

                while True:
                    success, image = cap.read()
                    if not success:
                        break
                    image_np = cv2.copyMakeBorder(image, 20, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    # Actual detection.
                    (boxes, scores, classes, num) = sess.run(
                      [detection_boxes, detection_scores, detection_classes, num_detections],
                      feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                      image_np,
                      np.squeeze(boxes),
                      np.squeeze(classes).astype(np.int32),
                      np.squeeze(scores),
                      category_index,
                      use_normalized_coordinates=True,
                      line_thickness=8)
                    # plt.figure(figsize=IMAGE_SIZE)
                    # plt.imshow(image_np)
                    # vis = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
                    # vis = image_np.astype(np.uint8)
                    cv2.imshow("Frame",image_np)
                    cv2.waitKey(1)
                    vout.write(image_np)
                cap.release()
                cv2.destroyAllWindows()
                writer.close()