import cv2 
import os
import pickle

def extract_frames(video,frames):
    if not os.path.exists(frames):
        os.mkdir(frames)
    cap = cv2.VideoCapture(video)
    count = 0
    while True:
        success, image = cap.read()
        if not success:
            break
        cv2.imwrite(frames + os.sep + "%s.jpg" % str(count).zfill(6), image)
        count += 1
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cap.release()


def make_video_from_list(out_vid_path, frames_list):
    if frames_list[0] is not None:
        img = cv2.imread(frames_list[0], True)
        print (frames_list[0])
        h, w = img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter(out_vid_path,fourcc, 2, (w, h), True)
        print("Start Making File Video:%s " % out_vid_path)
        print("%d Frames to Compress"%len(frames_list))
        for i in range(0,len(frames_list)):
            # if utils_image.check_image_with_pil(frames_list[i]):
            out.write(img)
            img = cv2.imread(frames_list[i], True)
        out.release()
        print("Finished Making File Video:%s " % out_vid_path)

def make_video_from_frames_dir(dir):
    f_list  = []
    out_vid_path = dir.split("\\")[-1] + ".avi"
    for f in os.listdir(dir):
        f_list.append(dir+os.sep+f)
    print(f_list)
    make_video_from_list(out_vid_path, f_list)


def vatic2gt(file):
    if os.path.isfile(file):
        outfile = open(file.split(".")[0]+"_out."+file.split(".")[-1],"w")
        print("here")
        with open(file,"r") as input:
            print("here")
            lines = input.readlines()
            for line in lines:
                splits = line.split(" ")
                print(line)
                print(splits)
                t_x, t_y, w, h = splits[1:5]
                b_y = str(int(t_y) + int(h))
                r_x = str(int(t_x) + int(w))
                outfile.write((",").join([t_x,t_y,r_x,t_y,t_x,b_y,r_x,b_y]))
                outfile.write("\n")
        outfile.close()
    return


def extract_gt(file = None ,dir = None):
    if file != None:
        vatic2gt(file)
    elif dir != None:
        for f in os.listdir(dir):
            if f.split(".")[-1] in ["xml"]:
                print(f)
                vatic2gt(dir + os.sep + f)
    return
if __name__ == '__main__':
    extract_frames("input/184651.avi", "input/184651")
    # make_video_from_frames_dir(dir = "D:\Datasets\scenarios\output\\bed_exit")
    # extract_gt(file = "C:\\Users\div_1\Downloads\\vatic-docker\data\output\\8.xml")

