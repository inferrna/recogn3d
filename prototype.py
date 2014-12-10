import numpy as np
import cv2
from time import clock
from PIL import Image
from math import log

class comparator():
    def __init__(self, sizey, sizex, dtype):
        self.emptyright = np.zeros((sizey+2, sizex+2), np.int16)
        self.emptyleft  = np.zeros((sizey+2, sizex+2), np.int16)
        self.szx = sizex
        self.szy = sizey
        maxd = np.iinfo(dtype).max*2
        allfill = np.empty((sizey, sizex), np.int16)
        allfill.fill(maxd)
        self.diffs = np.zeros((9, sizey, sizex), np.int16)
        self.diffs[0,0,:].fill(maxd)                                         #Top
        self.diffs[2,:,0].fill(maxd)                                         #Left
        self.diffs[1] = np.logical_or(self.diffs[0], self.diffs[2])*allfill #TopLeft
        self.diffs[4,sizey-1,:].fill(maxd)                                   #Botom
        self.diffs[3] = np.logical_or(self.diffs[2], self.diffs[4])*allfill #LeftBottom
        self.diffs[6,:,sizex-1].fill(maxd)                                   #Right
        self.diffs[5] = np.logical_or(self.diffs[4], self.diffs[6])*allfill #BottomRight
        self.diffs[7] = np.logical_or(self.diffs[6], self.diffs[0])*allfill #RightTop
        self.dirs = np.array([[-1,0],    #0 Top
                              [-1, -1],  #1 TopLeft
                              [0, -1],   #2 Left
                              [1, -1],   #3 LeftBottom
                              [1, 0],    #4 Bottom
                              [1, 1],    #5 BottomRight
                              [0, 1],    #6 Right
                              [-1, 1],   #7 RightTop
                              [0, 0]],   #8 Center
                              dtype=np.int16)+np.array([1, 1], dtype=np.int16)

    def compare(self, imgl, imgr):
        diffs = np.copy(self.diffs)
        emptyright = np.copy(self.emptyright)
        emptyright[1:self.szy+1, 1:self.szx+1] = imgr
        for i in range(0, len(self.dirs)):
            dy, dx = self.dirs[i]
            ey, ex = [self.szy+dy, self.szx+dx]
            diffs[i] += abs(emptyright[dy:ey,dx:ex] - imgl)
        #s[:,0,1].argmin()
        return diffs

dir1 = "screne/"
dir2 = "screne/resized/"
rnm = "rightballc2"
lnm = "leftballc2"
ext = ".png"

img0 = cv2.imread(dir1+rnm+ext)
cv_img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
img1 = cv2.imread(dir1+lnm+ext)
cv_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

oldy, oldx = cv_img0.shape

#vis = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
#cv2.imshow("test", vis)
#cv2.waitKey()
newy = oldy//2
newx = oldx//2

y = 1
x = 1

comparators = []
while x<oldx and y<oldy:
    x*=2
    y*=2
    comparators.append(comparator(y, x, cv_img0.dtype))
    szstr = '_'+str(x)+'x'+str(y)
    cv_img0rs = cv2.resize(cv_img0, (x, y), interpolation = cv2.INTER_AREA)
    #cv_img0rs = cv2.resize(cv_img0rs, (newx, newy), interpolation = cv2.INTER_NEAREST)
    cv_img1rs = cv2.resize(cv_img1, (x, y), interpolation = cv2.INTER_AREA) 
    #cv_img1rs = cv2.resize(cv_img1rs, (newx, newy), interpolation = cv2.INTER_NEAREST) 

    #vis = np.zeros((newy, 2*newx), cv_img0.dtype);
    #vis[:newy, :newx] = cv_img0rs
    #vis[:newy, newx:2*newx] = cv_img1rs
    #cv2.imwrite(dir2+'both'+szstr+ext, vis)

    res = comparators[-1].compare(cv_img0rs, cv_img0rs)
    print res
    exit()
