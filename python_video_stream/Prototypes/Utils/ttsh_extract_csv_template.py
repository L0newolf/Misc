import os

dir = "J:\\"
out_dir = "D:\\Conex"
patients = [32,58,17]
# beds = [2]

for folder in os.listdir(dir):
    print(folder)
    f = folder.split(" ")
    if len(f) >= 7 and int(f[2]) in patients: # and int(f[4]) in beds:
        out_file = out_dir + os.sep +  "P_" + str(int(f[2])) + ".csv"
        fl = open(out_file,"w")
        for year in os.listdir(dir + os.sep + folder):
            for month in os.listdir(dir + os.sep + folder + os.sep + year):
               for day in os.listdir(dir + os.sep + folder + os.sep + year + os.sep + month):
                    for v in os.listdir(dir + os.sep + folder + os.sep + year + os.sep + month + os.sep + day):
                        # print(dir + os.sep + folder + os.sep + year + os.sep + month + os.sep + day, v)
                        fl.write((",").join([folder, year, month, day, v]))
                        fl.write("\n")
        fl.close()

