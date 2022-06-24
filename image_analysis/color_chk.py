import cv2
import numpy as np
from color import convert as c

def main(path, cl,test=0):
    #path: img path
    #cl: cut list [start x, end x,start y, end y]
    img = cv2.imread(path)
    BGRlist=[]
    for i, c in enumerate(cl):
        cut = img[c[0]:c[1]+1,c[2]:c[3]+1,:]
        BGRlist.append(np.average(np.average(cut, axis = 0), axis = 0).tolist())
        if test!=0:
            trmimg=img.copy()
            cv2.rectangle(trmimg, (c[0],c[2]), (c[1], c[3]), (0, 255, 0),thickness=1)
            cv2.imwrite(path.replace('.','_trm'+str(i)+'.'), trmimg)

    return BGRlist

if __name__ in '__main__':
    path='test/test_99y_0.jpg'
    cl=[[60,100,40,80]]
    for BGR in main(path,cl,test=1):
        print(c.RGB_to_HLS(BGR[2]/255,BGR[1]/255,BGR[0]/255))