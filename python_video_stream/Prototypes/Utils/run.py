import configparser
import os

import cv2
import numpy as np

from Prototypes.Utils import tf_utils
from Prototypes.Object_Classifiers.classifier import Classifier, Methods
from Prototypes.object_detection import detect_object,get_moments
from Prototypes.Utils import image



class AlarmStatus:
    NO_ALARM,MULTIPLE_PERSON,BE_RED,BE_AMBER = range(0, 4)

class ObjectDetection:
    CV_INCEPTION,CV_SSD, SSD = range(0,3)

class BedExit_Detector(object):
    def __init__(self):
        config = configparser.ConfigParser()
        config.read("config.ini")
        self.config = config["DEFAULT"]
        self.overwrite_mask = self.config.getboolean("overwrite_mask")
        self.display = self.config.getboolean("display")
        self.points = []
        self.mask = None
        self.img_ext = ".jpg"
        self.np_ext = ".npy"
        self.read_bed_mask()
        self.line_color = (255,255,255)
        self.clf = Classifier()
        self.clf.set(method=Methods.MEAN)
        self.clf.model_file = self.config["model_file"]
        self.clf.load()
        tf_utils.graph_path = self.config["tf_model"]
        self.session = tf_utils.beginSession()
        self.alarm_color = {
            AlarmStatus.NO_ALARM : (0,255,0),
            AlarmStatus.MULTIPLE_PERSON : (0,127,0),
            AlarmStatus.BE_RED : (0,0,255),
            AlarmStatus.BE_AMBER : (0,69,255)
        }
        self.alarm_text = {
            AlarmStatus.NO_ALARM: "Patient OK",
            AlarmStatus.MULTIPLE_PERSON: "Multiple persons",
            AlarmStatus.BE_RED: "Red Alert",
            AlarmStatus.BE_AMBER: "Amber Alert"
        }
        self.object_classes = [1,2,3] #["person"]
        self.person_classes = [2]
        self.export_format = self.config["export_format"]
        self.prev_status = AlarmStatus.NO_ALARM
        self.current_status = AlarmStatus.NO_ALARM
        self.green_threshold = 0.80
        self.amber_thrsehold = 0.50
        self.mode = self.config["process_mode"]
        image.num = self.config.getint('rand_num')
        self.object_detection = self.config.getint('object_detection')
        self.patient_in_bed = True # 1: In Bed | 2: Not in Bed
        self.window_size = self.config.getint('window_size')
        self.status_buffer = np.zeros(self.window_size, dtype=int)


    def set_bed_mask(self,frame,window):
        self.mask, self.points = image.Utils.get_bed_mask_from_roi(frame, window)
        cv2.imwrite(self.config["mask_path"]+self.img_ext, self.mask)
        np.save(self.config["mask_path"] + self.np_ext, np.array([self.points]))

    def read_bed_mask(self):
        if os.path.isfile(self.config["mask_path"]+self.img_ext):
            self.mask = cv2.imread(self.config["mask_path"]+self.img_ext)
            self.points = np.load(self.config["mask_path"]+self.np_ext)

    def extract_objects(self,frame):

        if self.object_detection in [ObjectDetection.CV_INCEPTION,ObjectDetection.CV_SSD]:

            # creating the roi for image with threshold & bed-frame
            threshold, rect = detect_object(frame)
            cv2.imshow("threshold", threshold)
            cv2.bitwise_not(threshold, threshold)

            frames = []
            for r in rect:
                x, y, w, h = r
                if self.object_detection == ObjectDetection.CV_INCEPTION:
                    frames.append([frame[y:y + h, x:x + w]])
                elif self.object_detection == ObjectDetection.CV_SSD:
                    frames.append(frame[y:y + h, x:x + w])

            if self.object_detection == ObjectDetection.CV_INCEPTION:
                features = tf_utils.extract_features(frames)
                pred = self.clf.predict_MEAN(features)

            elif self.object_detection == ObjectDetection.CV_SSD:
                pred, boxes = tf_utils.ssd_predict(images= np.array(frames),sess= self.session,retrieve_max=True)

            remove_pred = []
            remove_rect = []
            for i in range(0, len(pred)):
                if pred[i] not in self.object_classes:
                # if False:
                    remove_pred.append(pred[i])
                    remove_rect.append(rect[i])
            for i in range(0, len(remove_pred)):
                pred.remove(remove_pred[i])
                rect.remove(remove_rect[i])

            return rect, pred, threshold

        elif self.object_detection == ObjectDetection.SSD:

            classes, boxes = tf_utils.ssd_predict([frame], self.session)
            # creating the roi for image with threshold & bed-frame

            threshold, rect = detect_object(frame.copy())
            cv2.imshow("threshold", threshold)
            cv2.bitwise_not(threshold, threshold)



            pred = []
            rect = []

            for i,r in enumerate(boxes[0]):
                if classes[0][i] not in self.object_classes:
                    continue
                ymin, xmin, ymax, xmax = r
                x = int(xmin* frame.shape[1])
                y = int(ymin *frame.shape[0])
                w = int(xmax* frame.shape[1]-x)
                h = int(ymax*frame.shape[0]-y)
                rect.append([x,y,w,h])
                pred.append(classes[0][i])

            return rect, pred, threshold
        return [],[],[]



    def update_status(self, bboxes, pred, threshold_image):

        sel = cv2.bitwise_and(cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY), threshold_image)
        person_count = 0
        for i in range(0,len(pred)):
            if pred[i] in self.person_classes:
                person_count += 1
        if person_count==0:
            return

        for i in range(0, len(bboxes)):
            x, y, w, h = bboxes[i]
            if pred[i] not in self.person_classes:
                continue
            # Create the basic black image
            mask = cv2.cvtColor(np.zeros(self.mask.shape, dtype="uint8"), cv2.COLOR_BGR2GRAY)

            # Draw a white, filled rectangle on the mask image
            cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)

            full_object = cv2.bitwise_and(threshold_image, mask)
            object_in_bed = cv2.bitwise_and(sel, mask)

            if cv2.countNonZero(full_object) > 0:
                percent_in_bed = cv2.countNonZero(object_in_bed) / cv2.countNonZero(full_object)
            else:
                percent_in_bed = 0


            # contours,moments,centroids,ellipsoids = get_moments(full_object)
            # areas = np.zeros(len(ellipsoids))
            # check = False
            # for idx,c in enumerate(ellipsoids):
            #     if c != None:
            #         check = True
            #         areas[idx] = (np.pi * c[1][0]* c[1][1]) / 4
            # if check == True:
            #     max_index = np.argmax(areas)
            # else:
            #     max_index = 0
            #     print("I'm here ")
            #
            # for idx,c in enumerate(centroids):
            #     m1,m2 = c
            #     cv2.circle(object_in_bed, (m1, m2), 2, (0, 0, 255), -1)
            #     if ellipsoids[idx] != None:
            #         cv2.ellipse(full_object, ellipsoids[idx], (0, 250, 0), 2)
            #
            # cv2.imshow("testing",full_object)
            #
            # c1,c2 = centroids[max_index]
            # print(self.mask[c1][c2],c1,c2)

            if  percent_in_bed == 0:
                if self.patient_in_bed == True:
                    self.patient_in_bed = False
            elif percent_in_bed >= self.amber_thrsehold: #self.mask[c1][c2][1] !=0:
                if self.patient_in_bed == False:
                    self.patient_in_bed = True

            flag = -1

            if person_count >= 2:
                flag = AlarmStatus.MULTIPLE_PERSON
            elif percent_in_bed >= self.green_threshold:
                flag = AlarmStatus.NO_ALARM
            elif percent_in_bed >= self.amber_thrsehold and self.patient_in_bed == True:
                flag = AlarmStatus.BE_AMBER
            elif pred[i] in self.object_classes and self.patient_in_bed == True:
                flag = AlarmStatus.BE_RED
            else:
                flag = AlarmStatus.NO_ALARM

            self.update_status_buffer(flag)

            if(self.prev_status!=self.current_status):
                self.output_file.write((",").join([self.date,self.time,str(int(self.frame_count)),self.alarm_text[self.current_status] + "\n"]))

    def update_status_buffer(self,flag): #Sliding window approach to minimize noise from
        self.prev_status = self.current_status

        for i in range(0,len(self.status_buffer)-1):
            self.status_buffer[i] = self.status_buffer[i+1]
        self.status_buffer[-1] = flag

        # Regularizing noise by picking maximum
        count = np.bincount(self.status_buffer)
        self.current_status  = np.argmax(count)


    def process_video(self,video_path):
        video = cv2.VideoCapture(video_path)
        date_list = video_path.split(".")[0].split(os.sep)[-4:-1]
        list.reverse(date_list)
        self.date = ("-").join(date_list)
        time_str = video_path.split(".")[0].split(os.sep)[-1]
        # self.time = datetime.datetime(100,1,1,int(time_str[0:2]),int(time_str[2:4]),int(time_str[4:]))
        self.time = (":").join([time_str[0:2],time_str[2:4],time_str[4:]])
        self.fps = video.get(cv2.CAP_PROP_FPS)

        # Temporary means to extract demo-videos
        # out_name = video_path.split(".")[0]
        #
        # out = cv2.VideoWriter(out_name  + "_out.avi", -1, 20.0, (384, 288))
        # print(out_name  + "_out.avi")
        self.frame_count = 0

        total = int((video.get(cv2.CAP_PROP_FRAME_COUNT) / 100) * 100)
        for i in range(0, total):
            res, frame = video.read()
            if res == False:
                break

            if len(self.points) == 0 or self.overwrite_mask == True:
                self.set_bed_mask(frame, "bed_frame")

            rect, pred, threshold = self.extract_objects(frame.copy())
            self.update_status(rect, pred, threshold)

            if self.display == True:
                cv2.polylines(frame, np.array([self.points]), False, self.line_color, 1)
                for i in range(0, len(rect)):
                    x, y, w, h = rect[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), self.alarm_color[self.current_status], 2)
                    cv2.putText(frame,  self.alarm_text[self.current_status]+ "_" + str(pred[i]), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, self.alarm_color[self.current_status])

                cv2.imshow("feed", frame)

                # cv2.imwrite(out_name + str(self.frame_count) + ".jpg", frame)
                # out.write(frame)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break

            self.frame_count += 1
        # out.release()
        video.release()
        cv2.destroyAllWindows()



    def get_video_paths(self):
        base = self.config["v_path"]
        patients = []
        paths = []

        if self.mode in ["d","file","m","y", "p"]:
            patients.append(self.config["v_patient"])
        else:
            patients = os.listdir(base)

        for patient in patients:
            p_patient = base + os.sep + patient
            years = []
            if self.mode in ["d","file","m","y"]:
                years.append(self.config["v_year"])
            else:
                years = os.listdir(p_patient)

            for year in years:
                p_year = p_patient + os.sep + year
                months = []
                if self.mode in ["d","file","m"]:
                    months.append(self.config["v_month"])
                else:
                    months = os.listdir(p_year)


                for month in months:
                    days = []
                    p_month = p_year + os.sep + month
                    if self.mode in ["d","file"]:
                        days.append(self.config["v_day"])
                    else:
                        days = os.listdir(p_month)

                    for day in days:
                        p_day = p_month + os.sep + day
                        files = []
                        if self.mode == "file":
                            files.append(self.config["v_file"])
                        else:
                            for f in os.listdir(p_day):
                                ## Add only those files that have been recording within the specified time frame
                                ## To process all the files in a day, set start_hour = 00 and end_hour = 23
                                if int(f[0:2])>= int(self.config["v_start_hour"]) and int(f[0:2]) <= int(self.config["v_end_hour"]):
                                    files.append(f)


                        for file in files:
                            paths.append(p_day + os.sep + file)
        return paths

    def run(self):
        list_paths = self.get_video_paths()
        list.sort(list_paths)
        print(list_paths)
        # list.reverse(list_paths)
        file_count = 0
        self.output_file = open(self.config["out_file"], "w")
        for video_path in list_paths:
            if list_paths.index(video_path) % 12 == 0:
                file_count += 1
                self.output_file.close()
                self.output_file = open(self.config["out_file"] + "_%d"%file_count + "." + self.config["export_format"], "w")
            print(video_path)
            self.process_video(video_path)

        self.output_file.close()

    def analyze_results(self):
        gt = open(self.config["ground_truth"])
        lines = gt.readlines()
        gt.close()
        gt_data = dict()
        for line in lines:
            line = line.split(",")
            patient = line[0].split(" ")[2]
            date = ("-").join([line[3].zfill(2),line[2].zfill(2),line[1][-2:].zfill(2)])
            file = int(line[4].split(".")[0])
            if patient not in gt_data.keys():
                gt_data[patient] = dict()
            if date not in gt_data[patient].keys():
                gt_data[patient][date] = set()
            gt_data[patient][date].add(file)

        patient = "003" # Todo: Fix this
        pred_path = self.config["pred_path"]
        lines = []
        for fl in os.listdir(pred_path):
            lines.extend(open(pred_path + os.sep + fl).readlines())

        pred_data = dict()
        for line in lines:
            line = line.strip().split(",")
            date = line[0]
            file = int(("").join(line[1].split(":")))
            tag = line[3]
            if tag in ["Red Alert","Amber Alert"]:
                if patient not in pred_data.keys():
                    pred_data[patient] = dict()
                if date not in pred_data[patient].keys():
                    pred_data[patient][date] = set()
                pred_data[patient][date].add(file)
        far = 0
        frr = 0
        predicted = 0
        correct = 0
        true_gt = 0
        for patient in gt_data.keys():
            for date in gt_data[patient].keys():
                for t in gt_data[patient][date]:
                    found = False
                    if patient in pred_data.keys():
                        if date in pred_data[patient].keys():
                            if t in pred_data[patient][date]:
                                found = True
                        if found == False:
                            frr += 1
                            print(date,t)
                        true_gt += 1
        for patient in pred_data.keys():
            for date in pred_data[patient].keys():
                for t in pred_data[patient][date]:
                    found = False
                    if patient in gt_data.keys():
                        if date in gt_data[patient].keys():
                            if t in gt_data[patient][date]:
                                found = True
                    if found == False:
                        far += 1
                    else:
                        correct += 1
                    predicted +=1
        print(far,frr,predicted,correct,true_gt)
        print(gt_data)
        print(pred_data)

if __name__=="__main__":
    clf = BedExit_Detector()
    clf.run()
    # clf.analyze_results()