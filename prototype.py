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
        self.diffs[0,0,:].fill(maxd)                                         #0 Top
        self.diffs[2,:,0].fill(maxd)                                         #2 Left
        self.diffs[1] = np.logical_or(self.diffs[0], self.diffs[2])*allfill  #1 TopLeft
        self.diffs[4,sizey-1,:].fill(maxd)                                   #4 Botom
        self.diffs[3] = np.logical_or(self.diffs[2], self.diffs[4])*allfill  #3 LeftBottom
        self.diffs[6,:,sizex-1].fill(maxd)                                   #6 Right
        self.diffs[5] = np.logical_or(self.diffs[4], self.diffs[6])*allfill  #5 BottomRight
        self.diffs[7] = np.logical_or(self.diffs[6], self.diffs[0])*allfill  #7 RightTop, 8 Center is full of zeroes
        self.dirs = np.array([[-1,  0],    #0 Top
                              [-1, -1],    #1 TopLeft
                              [ 0, -1],    #2 Left
                              [ 1, -1],    #3 LeftBottom
                              [ 1,  0],    #4 Bottom
                              [ 1,  1],    #5 BottomRight
                              [ 0,  1],    #6 Right
                              [-1,  1],    #7 RightTop
                              [ 0,  0]],   #8 Center
                              dtype=np.int16)

    def compare(self, imgl, imgr, offsets):
        diffs = np.copy(self.diffs)#.reshape(self.diffs.shape+[2, 2])
        emptyright = np.copy(self.emptyright)
        emptyright[1:self.szy+1, 1:self.szx+1] = imgr
        dirs = self.dirs+np.array([1, 1], dtype=np.int16)
        for i in range(0, len(dirs)):
            dy, dx = dirs[i]                        #Directions to shift (normalized)
            ey, ex = [self.szy+dy, self.szx+dx]     #Ends
            for gy in range(0, self.szy):
                for gx in range(0, self.szx):
                    ox, oy = offsets[gy, gx] #Previvious offsets (scaled)
                    #print "imgl[gy, gx] is", imgl[gy, gx]
                    #print "emptyright[0,0] is", emptyright[0,0]
                    #print "dy,(dy+gy+oy), dx,(dx+gx+ox) is", dy,(dy+gy+oy), dx,(dx+gx+ox)
                    #print "emptyright[dy:(dy+gy+oy),dx:(dx+gx+ox)] is", emptyright[dy:(dy+gy+oy),dx:(dx+gx+ox)]
                    #print "difference is", emptyright[dy:(dy+gy+oy),dx:(dx+gx+ox)] - imgl[gy, gx]
                    diffs[i, gy, gx] += abs(emptyright[(dy+gy+oy),(dx+gx+ox)] - imgl[gy, gx]) #Needs changes
        sm = diffs.argmin(axis=0)          #To find best offset indexes
        offs = self.dirs[sm]+offsets   #Find new offsets and append the olds
        ###To scale by nearest###
        #s2[::2,::2,:] = s 
        #s2[1::2] = s2[::2] 
        #s2[:,1::2,:] = s2[:,::2,:] 
        print "------Diffs==\n", diffs
        return offs.repeat(2,axis=0).repeat(2,axis=1)

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
offsets = np.zeros((2,2,2)).astype(np.int16)
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

    offsets = comparators[-1].compare(cv_img0rs, cv_img0rs, offsets)
    print "----Offsets is\n", offsets
    exit()
