import os
import cv2
import matplotlib.pyplot as plt
import background_segmentation as bs

dir = os.getcwd() + "\\videos\\17\\05\\24"

timeStamps = []
percent_activity = []

def analyze_video(vfile):
    activity = bs.backgroundSubtract(vfile)
    timeStamps.append(vfile.split("\\")[-1].split(".")[0])
    percent_activity.append(activity)
    print(activity)


for f in os.listdir(dir):
    filepath = os.path.join(dir, f)
    if os.path.isfile(filepath) and f.split(".")[-1]=="avi":
        analyze_video(filepath)


plt.plot(timeStamps,percent_activity)
plt.show()