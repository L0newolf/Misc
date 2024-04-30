import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.tools import freeze_graph
import numpy as np
import os
import pickle
from Prototypes.Utils.image import Utils
#from object_detection.utils import dataset_util

import cv2

# graph_path = os.getcwd() + "/../models/inception_v3_ssd/tensorflow_inception_graph.pb"

def create_graph(model_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def beginSession():
    print(graph_path)
    detection_graph = create_graph(graph_path)
    sess = tf.Session(graph=detection_graph)
    return sess


def ssd_predict(images, sess = None, retrieve_max=False, min_confidence=0.6):
    if sess is None:
        sess = beginSession()

    image_tensor =  sess.graph.get_tensor_by_name('image_tensor:0')
    detection_boxes =  sess.graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = sess.graph.get_tensor_by_name('detection_scores:0')
    detection_classes = sess.graph.get_tensor_by_name('detection_classes:0')
    num_detections = sess.graph.get_tensor_by_name('num_detections:0')
    predictions = []
    bboxes = []
    for image_np in images:
        image_np_expanded = np.expand_dims(image_np, axis=0)
        (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_np_expanded})
        if retrieve_max == True:
            max_index = np.argmax(scores[0])
            if scores[0][max_index] >= min_confidence:
                predictions.append(int(classes[0][max_index]))
                bboxes.append(boxes[0][max_index])
        else:
            cls = []
            box = []
            for i,c in enumerate(classes[0]):
                if scores[0][i] >= min_confidence:
                    cls.append(c)
                    box.append(boxes[0][i])
            predictions.append(cls)
            bboxes.append(box)
    return predictions, bboxes



def extract_features(image_paths, verbose=False):
    """
    extract_features computed the inception bottleneck feature for a list of images

    image_paths: array of image path
    return: 2-d array in the shape of (len(image_paths), 2048)
    """
    feature_dimension = 2048
    features = np.empty((len(image_paths), feature_dimension))

    with tf.Session() as sess:
        flattened_tensor = sess.graph.get_tensor_by_name('pool_3:0') #pool_3:0

        for i, image_path in enumerate(image_paths):
            if verbose:
                print('Processing %s...' % (image_path))

            if isinstance(image_path,list):
                image_data = Utils.cv2_tf_decoded_image(image_path[0])
            else:
                if not gfile.Exists(image_path):
                    tf.logging.fatal('File does not exist %s', image_path)
                image_data = gfile.FastGFile(image_path, 'rb').read()
            feature = sess.run(flattened_tensor, {
                'DecodeJpeg/contents:0': image_data
            })
            features[i, :] = np.squeeze(feature)

    return features

def get_incep_feat_from_dir(input_dir,output_file):
    if not os.path.exists(input_dir):
        print("The input directory does not exist. Please provide a valid training directory")
        return
    data = dict()
    images_list = []
    image_map = dict()
    for object_dir in os.listdir(input_dir):
        for image in os.listdir(input_dir + os.sep + object_dir):
            id = image.split(".")[0]
            image_map[len(images_list)] = id
            data[id] = dict()
            data[id]["label"] = object_dir
            images_list.append(input_dir + os.sep + object_dir + os.sep + image)
    create_graph(graph_path)
    feat = extract_features(images_list)
    for i in range(0,len(feat)):
        data[image_map[i]]["path"] = images_list[i]
        data[image_map[i]]["feat"] = feat[i]
    pickle.dump(data, open(output_file, "wb"))

def get_incep_feat(list,output_file):
    data = dict()
    create_graph(graph_path)
    feat = extract_features(list)
    for i in range(0, len(feat)):
        data[i] = dict()
        data[i]["label"] = list[i][1]
        data[i]["feat"] = feat[i]
    pickle.dump(data, open(output_file, "wb"))

def create_tf_record(list, output_path):
    print("creating tf record ", output_path)
    writer = tf.python_io.TFRecordWriter(output_path)
    for tf_example in list:
        writer.write(tf_example.SerializeToString())
    writer.close()

"""
def create_tf_example(example,labels_dict):
    frame = example[0]
    height, width, channels = frame.shape

    # filename = ("%s.jpg" %i).encode()
    encoded_image_data = Utils.cv2_tf_encoded_image(frame)  # Encoded image bytes
    image_format = ('jpeg').encode()  # b'jpeg' or b'png'

    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    for label,box in example[1]:
        classes_text.append(label.encode())
        classes.append(int(labels_dict[label]))
        xmin,ymin,xmax,ymax = box
        xmins.append(float(xmin/width))
        ymins.append(float(ymin/width))
        xmaxs.append(float(xmax/height))
        ymaxs.append(float(ymax/height))


    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        # 'image/filename': dataset_util.bytes_feature(filename),
        # 'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label':dataset_util.int64_list_feature(classes),
    }))
    return tf_example
"""

def inspect_nodes(graph_path):
    g = tf.GraphDef()
    g.ParseFromString(open(graph_path, 'rb').read())
    d = [n for n in g.node if True or n.name.find('output') != -1]  # same for output or any other node you want to make sure is ok
    print(d)

if __name__=="__main__":
    graph_path = "D:\\Conex\\PreSage\\Prototypes\\Dataset_10h_wk1\\model_gpu_12K\\frozen_model\\frozen_inference_graph.pb"
    inspect_nodes(graph_path)