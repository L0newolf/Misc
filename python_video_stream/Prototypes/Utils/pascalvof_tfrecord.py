import os
import sys
import random

import numpy as np
import tensorflow as tf
from object_detection.utils import dataset_util
import xml.etree.ElementTree as ET

# Original dataset organisation.
DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'JPEGImages/'

# TFRecords convertion parameters.
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 1750
VOC_Label = {
    'n03920384':['gauze',1],
    'n02786058':['band aid',2],
    'n04254120':['sanitizer',3],
    'n04317175':['stethescope',4],
    'n04376876':['syringe',5]
}

def _process_image(directory, name):
    """Process a image and annotation file.
    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    filename = directory + DIRECTORY_IMAGES + name + '.JPEG'

    with tf.gfile.GFile(filename, 'rb') as fid:
        image_data = fid.read()

    # Read the XML annotation file.
    filename = os.path.join(directory, DIRECTORY_ANNOTATIONS, name + '.xml')

    tree = ET.parse(filename)
    root = tree.getroot()

    # Image shape.
    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]
    # Find annotations.
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(VOC_Label[label][1]))
        labels_text.append(VOC_Label[label][0].encode('ascii'))

        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        bbox = obj.find('bndbox')
        bboxes.append((float(bbox.find('ymin').text) / float(shape[0]),
                       float(bbox.find('xmin').text) / float(shape[1]),
                       float(bbox.find('ymax').text) / float(shape[0]),
                       float(bbox.find('xmax').text) / float(shape[1])
                       ))
        print(bboxes)
    return image_data, shape, bboxes, labels, labels_text, difficult, truncated


def _convert_to_example(image_data, labels, labels_text, bboxes, shape,
                        difficult, truncated):
    """Build an Example proto for an image example.
    Args:
      image_data: string, JPEG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned

    image_format = ('jpeg').encode()
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(shape[0]),
            'image/width': dataset_util.int64_feature(shape[1]),
            'image/channels': dataset_util.int64_feature(shape[2]),
            'image/shape': dataset_util.int64_list_feature(shape),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
            'image/object/class/label': dataset_util.int64_list_feature(labels),
            'image/object/class/text': dataset_util.bytes_list_feature(labels_text),
            # 'image/object/bbox/difficult': dataset_util.int64_list_feature(difficult),
            # 'image/object/bbox/truncated': dataset_util.int64_list_feature(truncated),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/encoded': dataset_util.bytes_feature(image_data)}))
    return example


def _add_to_tfrecord(dataset_dir, name, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.
    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    image_data, shape, bboxes, labels, labels_text, difficult, truncated = \
        _process_image(dataset_dir, name)
    example = _convert_to_example(image_data, labels, labels_text,
                                  bboxes, shape, difficult, truncated)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def run(dataset_dir, output_dir, name='voc_train', shuffling=False):
    """Runs the conversion operation.
    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    # Dataset filenames, and shuffling.
    path = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS)
    objects = sorted(os.listdir(path))
    filenames = []
    for obj in objects:
        filenames.extend(sorted(os.listdir(path + os.sep + obj)))
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)

    # Process dataset files.
    i = 0
    fidx = 0
    while i < len(filenames):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < SAMPLES_PER_FILES:
                print(i)
                filename = filenames[i]
                img_name = filename.split("_")[0] + os.sep + filename[:-4]
                if not os.path.exists(dataset_dir + os.sep + DIRECTORY_IMAGES + img_name + ".JPEG"):
                    print(dataset_dir + os.sep + DIRECTORY_IMAGES + img_name + ".JPEG")
                    sys.stdout.write('\r>> could not find image %d/%d \n' % (i + 1, len(filenames)))
                    i += 1
                else:
                    sys.stdout.write('\r>> Converting image %d/%d \n' % (i + 1, len(filenames)))
                    _add_to_tfrecord(dataset_dir, img_name, tfrecord_writer)
                    j += 1
                    i += 1
                sys.stdout.flush()


            fidx += 1

    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the Pascal VOC dataset!')

run("D:\Datasets\Medical\\","D:\Datasets\Medical",shuffling=True)