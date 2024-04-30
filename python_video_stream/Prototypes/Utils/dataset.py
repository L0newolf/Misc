from loader import Loader
from image import Utils
import xml.etree.ElementTree as ET
import configparser
import ast
import os
import cv2
import random
import pickle
import tf_utils

class MODE:
    SAVE, SAVE_PASCALVOF, TF_EXTRACT, TF_RECORD = range(0,4)

class Dataset(object):
    def __init__(self):
        config = configparser.ConfigParser()
        config.read("config.ini")
        self.config = config["DEFAULT"]
        self.train_features = None
        self.train_labels = None
        self.evaluation_features = None
        self.evaluation_labels = None
        self.data = None
        self.eval_size = self.config.getfloat("eval_size")
        self.dataset_folder = self.config["dataset_folder"]
        self.images = self.dataset_folder + os.sep + self.config["images"]
        self.annotations = self.dataset_folder + os.sep + self.config["annotations"]
        if not os.path.exists(self.images):
            os.mkdir(self.images)
        if not os.path.exists(self.annotations):
            os.mkdir(self.annotations)
        self.eval_dir = self.dataset_folder + os.sep + "eval"
        self.train_dir = self.dataset_folder + os.sep + "train"
        self.labels_dict = dict()
        self.num_classes = 1
        self.ignore_classes = ast.literal_eval(self.config["ignore_obj"])
        self.object_counts = dict()
        self.mode = self.config.getint("mode")
        self.processed = set()
        self.fir = int(100/self.config.getint("fir"))               #Frame Inclusion Rate (%)
        if os.path.exists(self.config["processed_list"]):
            self.processed = pickle.load(open(self.config["processed_list"],"rb"))

    def extract_data(self,_frame,_objects,_prefix):
        if self.mode in [MODE.SAVE,MODE.TF_EXTRACT]:
            for obj_id in _objects:
                attr = _objects[obj_id]
                if attr['outside'] == '0':
                    image = Utils.crop_frame(frame=_frame, box=(
                    int(attr['xtl']), int(attr['ytl']), int(attr['xbr']), int(attr['ybr'])))
                    label = attr['label']
                    if label in self.ignore_classes:
                        continue
                    if label not in self.labels_dict.keys():
                        self.labels_dict[label] = self.num_classes
                        self.num_classes += 1
                        self.object_counts[label] = 0
                    self.object_counts[label] += 1
                    self.samples.append([image, label])
        elif self.mode in [MODE.TF_RECORD,MODE.SAVE_PASCALVOF]:
            objects = []
            for obj_id in _objects:
                attr = _objects[obj_id]
                if attr['outside'] == '0':
                    box = (int(attr['xtl']), int(attr['ytl']), int(attr['xbr']), int(attr['ybr']))
                    label = attr['label']
                    if label in self.ignore_classes:
                        continue
                    if label not in self.labels_dict.keys():
                        self.labels_dict[label] = self.num_classes
                        self.num_classes += 1
                        self.object_counts[label] = 0
                        if not os.path.exists(self.annotations + os.sep + label):
                            os.mkdir(self.annotations + os.sep + label)
                        if not os.path.exists(self.images + os.sep + label):
                            os.mkdir(self.images + os.sep + label)
                    self.object_counts[label] += 1
                    if self.mode == MODE.SAVE_PASCALVOF:
                        obj_prefix = self.images + os.sep + label + os.sep + _prefix + "_" + str(self.object_counts[label])
                        ## write the image where the object is found to the corresponding obj path
                        cv2.imwrite(obj_prefix + ".jpg",_frame)
                        ## create annotations file
                        self.save_voc_annotations(_prefix, box, label, _frame.shape)
                    objects.append([label, box])
            if self.mode == MODE.TF_RECORD:
                self.samples.append(tf_utils.create_tf_example([_frame, objects], self.labels_dict))

    ## Writes data to .xml file in pascal_voc format
    def save_voc_annotations(self, _prefix,bbox,label,imgDim):
        ## Creating the basic structure
        root = ET.Element('annotation')
        folder = ET.SubElement(root,'folder')
        filename = ET.SubElement(root,'filename')
        size = ET.SubElement(root,'size')

        width = ET.SubElement(size,'width')
        height = ET.SubElement(size,'height')
        depth= ET.SubElement(size, 'depth')
        object = ET.SubElement(root,'object')
        name = ET.SubElement(object,'name')
        box = ET.SubElement(object,'bndbox')
        xmin = ET.SubElement(box,'xmin')
        ymin = ET.SubElement(box, 'ymin')
        xmax = ET.SubElement(box, 'xmax')
        ymax = ET.SubElement(box, 'ymax')

        ## Adding the appropriate values
        folder.text = label
        if len(imgDim) == 2:
            d = 1
            w, h = imgDim
        else:
            w,h,d = imgDim
        width.text = str(w)
        height.text = str(h)
        depth.text = str(d)
        name.text = label
        xm,ym,xmx,ymx = bbox
        xmin.text = str(xm)
        ymin.text = str(ym)
        xmax.text = str(xmx)
        ymax.text = str(ymx)
        _path = self.annotations + os.sep + label + os.sep + _prefix + "_" + str(self.object_counts[label])
        filename.text = _path.split(os.sep)[-1]
        ## Saving the file
        tree = ET.ElementTree(root)
        tree.write(_path+".xml")

    def createDataset(self):
        l = Loader()
        l.use_attributes_as_obj = self.config.getboolean("use_attributes_as_obj")
        self.samples = []
        self.data = l.data
        keys = list(self.data.keys())
        random.shuffle(keys)
        for v_id in keys:
            if v_id not in self.processed:
                if not os.path.exists(self.data[v_id]["path"]):
                    print(self.data[v_id]["path"] + " doesn't exist")
                    continue
                else:
                    print("processsing" + self.data[v_id]["path"] )
                    self.processed.add(v_id)
                video = cv2.VideoCapture(self.data[v_id]["path"])

                for f_id in self.data[v_id]["frames"].keys():
                    if int(f_id) % self.fir == 0:
                        frame = Utils.extract_frame(video=video, frame_number=int(f_id))
                        _objects = self.data[v_id]["frames"][f_id]
                        _prefix = v_id + "_" + f_id
                        self.extract_data(frame,_objects,_prefix)

        print(self.object_counts)

        #Shuffling samples obtained to create train/eval splits
        random.shuffle(self.samples)
        to_train = int((1 - self.eval_size) * len(self.samples))
        train_list = self.samples[:to_train]
        eval_list = self.samples[to_train:]
        if  self.mode == MODE.SAVE:
            self.save(train_list, self.train_dir)
            self.save(eval_list, self.eval_dir)
        elif  self.mode == MODE.TF_EXTRACT:
            tf_utils.get_incep_feat(train_list, self.train_dir + ".pickle")
            tf_utils.get_incep_feat(eval_list, self.eval_dir + ".pickle")
        elif  self.mode == MODE.TF_RECORD:
            tf_utils.create_tf_record(train_list, self.train_dir + "_tf.record")
            tf_utils.create_tf_record(eval_list, self.eval_dir + "_tf.record")

    def save(self, list, dir):
        if not os.path.exists(dir):
            os.mkdir(dir)
        count = 0
        for item in list:
            obj_dir = dir + os.sep + item[1]
            if not os.path.exists(obj_dir):
                os.mkdir(obj_dir)
            cv2.imwrite(obj_dir + os.sep + "%d.jpg" % count ,item[0])
            count +=1


if __name__ == "__main__":
   l = Dataset()
   l.createDataset()





