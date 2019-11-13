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
filenames = ['plain_data/record_0.avi', 'plain_data/record_1.avi','plain_data/record_2.avi','plain_data/record_3.avi','plain_data/record_4.avi', 'plain_data/record_5.avi', 'plain_data/record_9.avi', 'plain_data/record_7.avi', 'plain_data/record_8.avi', 'plain_data/record_10.avi', 'plain_data/record_11.avi','plain_data/record_12.avi']
labels = [1,0,1,2,3,1,1,0,2,0,2,3]
filenames += ['plane/record_0.avi','plane/record_1.avi','plane/record_2.avi','plane/record_3.avi','plane/record_4.avi']
labels += [2,1,0,3,3]
i=0
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
            edged=cv2.Canny(frame,50,150)
            contours, hierarchy=cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(frame,contours,-1,(0,0,0),3)
            blur = cv2.blur(frame,(5,5))
            gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
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
