import cv2
import csv
import numpy as np

# previous	0
# next 		1
# stop		2
# junk		3

outfilename = 'labels2.csv'
data = [[]]
fps_needed = 10
filenames = ['bitmask/record_0.avi','bitmask/record_1.avi','bitmask/record_2.avi','bitmask/record_3.avi']
labels = [2,1,0,3]
filenames += ['gcl_rand/record_1.avi','gcl_rand/record_2.avi','gcl_rand/record_3.avi','gcl_rand/record_4.avi','gcl_rand/record_5.avi','gcl_rand/record_6.avi','gcl_rand/record_7.avi']
labels += [3,2,0,1,0,1,2]
filenames += ['gcl2/record_0.avi','gcl2/record_1.avi','gcl2/record_2.avi','gcl2/record_3.avi','gcl2/record_4.avi','gcl2/record_5.avi','gcl2/record_6.avi','gcl2/record_7.avi']
labels += [2,0,1,3,3,2,0,1]
filenames += ['gcl3/record_0.avi','gcl3/record_1.avi','gcl3/record_2.avi','gcl3/record_3.avi','gcl3/record_4.avi']
labels += [1,1,2,0,3]
kernel = np.ones((3,3),np.uint8)
i=0
lower = np.array([108, 23, 82], dtype = "uint8")
upper = np.array([179, 255, 255], dtype = "uint8")
for count in range(0,len(filenames)):
    filename = filenames[count]
    label = labels[count]
    cap= cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("video fps = ", fps)
    mod = fps/fps_needed
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if (i%mod == 0):
            width = 50
            height = 50
            dim = (width, height)
            nemo = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv_nemo, lower, upper)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
            hsv_d = cv2.morphologyEx(mask,cv2.MORPH_OPEN, kernel)
            resized = cv2.resize(hsv_d, dim, interpolation = cv2.INTER_AREA)
            out_fname = 'out/img'+str(i)+'.jpg'
            cv2.imwrite(out_fname,resized)
            data.append([out_fname,str(label)])
        i+=1
    cap.release()
    cv2.destroyAllWindows()

np.random.shuffle(data)
print(len(data))
wtr = csv.writer(open (outfilename, 'w'), delimiter=',', lineterminator='\n')
for x in data : wtr.writerow (x)
