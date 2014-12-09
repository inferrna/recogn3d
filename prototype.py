import numpy as np
import cv2
from time import clock
from PIL import Image
from math import log

dir1 = "screne/"
dir2 = "screne/resized/"
rnm = "rightballc2"
lnm = "leftballc2"
ext = ".png"

img0 = cv2.imread(dir1+rnm+ext)
cv_img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
img1 = cv2.imread(dir1+lnm+ext)
cv_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

y, x = cv_img0.shape

#vis = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
#cv2.imshow("test", vis)
#cv2.waitKey()
newx = x//2
newy = y//2

class comparator():
    def __init__(self, sizex, sizey, dtype):
        self.emptyright = np.zeros((sizey+2, sizex+2), dtype)
        self.emptyleft  = np.zeros((sizey+2, sizex+2), dtype)
        self.szx = sizex
        self.szy = sizey
        allfill = np.empty((sizey, sizex), np.int16)
        allfill.fill(255)
        self.diffs = np.zeros((9, sizey, sizex), np.int16)
        self.diffs[0,0,:].fill(255)                                         #Top
        self.diffs[2,:,0].fill(255)                                         #Left
        self.diffs[1] = np.logical_or(self.diffs[0], self.diffs[2])*allfill #TopLeft
        self.diffs[4,sizey-1,:].fill(255)                                   #Botom
        self.diffs[3] = np.logical_or(self.diffs[2], self.diffs[4])*allfill #LeftBottom
        self.diffs[6,:,sizex-1].fill(255)                                   #Right
        self.diffs[5] = np.logical_or(self.diffs[4], self.diffs[6])*allfill #BottomRight
        self.diffs[7] = np.logical_or(self.diffs[6], self.diffs[0])*allfill #RightTop
        self.dirs = np.array([[-1,0],   #0 Top
                             [-1, -1],  #1 TopLeft
                             [0, -1],   #2 Left
                             [1, -1],   #3 LeftBottom
                             [1, 0],    #4 Bottom
                             [1, 1],    #5 BottomRight
                             [0, 1],    #6 Right
                             [-1, 1],   #7 RightTop
                             [0, 0]],   #8 Center
                             dtype=np.int16)+np.array([1, 1], dtype=np.int16)

    def compare(imgr, imgl):
        diffs = np.copy(self.diffs)
        emptyright = np.copy(self.emptyright)
        emptyright[1:self.szy, 1:self.szx] = imgr
        for i in range(0, len(self.dirs)):
            dy, dx = self.dirs[i]
            ey, ex = [self.szy+dy, self.szx+dx]
            diffs[i] += emptyright[dy:ey,dx:ex] - imgl



while x>2 and y>2:
    x//=2
    y//=2
    szstr = '_'+str(x)+'x'+str(y)
    cv_img0rs = cv2.resize(cv_img0, (x, y), interpolation = cv2.INTER_AREA)
    #cv_img0rs[0,1] = 0
    cv_img0rs = cv2.resize(cv_img0rs, (newx, newy), interpolation = cv2.INTER_NEAREST)
    cv_img1rs = cv2.resize(cv_img1, (x, y), interpolation = cv2.INTER_AREA) 
    cv_img1rs = cv2.resize(cv_img1rs, (newx, newy), interpolation = cv2.INTER_NEAREST) 
    #cv2.imwrite(dir2+rnm+szstr+ext, cv_img0rs)
    #cv2.imwrite(dir2+lnm+szstr+ext, cv_img1rs)
    vis = np.zeros((newy, 2*newx), cv_img0.dtype);
    vis[:newy, :newx] = cv_img0rs
    vis[:newy, newx:2*newx] = cv_img1rs
    cv2.imwrite(dir2+'both'+szstr+ext, vis)
