import os, sys
import getopt
import xml.etree.ElementTree as ET

class Loader(object):
    def __init__(self, argv = ''):
        # TODO: Fix the defaults to be within the project_package
        self.allowed_ext = [".avi", ".mp4", ".wmv", ".flv", ".mov"]
        self.annotations_folder = "D:\\Datasets\\TTSH_Annotated"
        self.video_dict_f = self.annotations_folder + os.sep + "video_dict.txt"
        self.video_folder = "D:\\Datasets\\TTSH_Videos"
        self.use_attributes_as_obj = True
        try:
            opts, args = getopt.getopt(argv, "ha:v:d:", ["help","v_in=", "a_in=", "v_dict="])
        except getopt.GetoptError:
            print('loader.py -a <annotation_folder> -v <video_folder> -d <video_dict> ')
            print('loader.py --a_in <annotation_folder> --v_in <video_folder> --dict <video_dict> ')
            sys.exit(2)
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                print('loader.py -a <annotation_folder> -v <video_folder> -d <video_dict> ')
                sys.exit()
            elif opt in ("-a", "--a_in"):
                self.annotations_folder = arg
                self.video_dict_f = self.annotations_folder + os.sep + "video_dict.txt"
            elif opt in ("-v", "--v_in"):
                self.video_folder = arg
            elif opt in ("-d", "--v_dict"):
               self.video_dict_f = arg

    def read(self):
        if os.path.exists(self.video_dict_f):
            self.video_dict = dict()
            f = open(self.video_dict_f)
            for line in f.readlines():
                id, path = line.split(",")
                self.video_dict[id] = dict()
                self.video_dict[id]["path"] = path.replace("/root/vatic/data/frames_in",self.video_folder).strip() + ".avi"
                self.video_dict[id]["frames"] = self.load_xml(id)
            f.close()

    def load_xml(self,id):
        frames = dict()
        f = self.annotations_folder + os.sep + id + ".xml"
        if os.path.exists(f):
            tree = ET.parse(f)
            root = tree.getroot()
            for obj in root.findall("track"):
                obj_id = obj.get("id").strip()
                label = obj.get("label").strip()
                attributes = obj[0].find("attribute")
                if attributes != None and self.use_attributes_as_obj == True:
                    label = label + "_" + attributes.text.strip()
                for child in obj:
                    frame_id = child.get("frame")
                    del child.attrib['frame']
                    child.attrib['label'] = label
                    if frame_id not in frames.keys():
                        frames[frame_id] = dict()
                    frames[frame_id][obj_id] = child.attrib
        return frames

    def __getattr__(self, data):
        self.read()
        return self.video_dict

if __name__ == "__main__":
   l = Loader(sys.argv[1:])
   l.data