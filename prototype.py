import numpy as np
import cv2
from time import clock
from PIL import Image, ImageDraw
from math import log
from random import randint

#def blockslice(arr, sy, sx):
#u.transpose(1,0).reshape((2,2,4,)).transpose(0,2,1).reshape((2,2,2,2,))

class comparator():
    def __init__(self, sizey, sizex, dtype):
        self.emptyright = np.zeros((sizey+2, sizex+2), np.int16)
        self.emptyleft  = np.zeros((sizey+2, sizex+2), np.int16)
        self.szx = sizex
        self.szy = sizey
        maxd = np.iinfo(dtype).max*4
        allfill = np.empty((sizey, sizex), np.int32)
        allfill.fill(maxd)
        self.diffs = np.zeros((9, sizey, sizex), np.int32)
        self.diffs[1,0,:].fill(maxd)                                         #1 Top
        self.diffs[3,:,0].fill(maxd)                                         #3 Left
        self.diffs[2] = np.logical_or(self.diffs[0], self.diffs[2])*allfill  #2 TopLeft
        self.diffs[5,sizey-1,:].fill(maxd)                                   #5 Botom
        self.diffs[4] = np.logical_or(self.diffs[2], self.diffs[4])*allfill  #4 LeftBottom
        self.diffs[7,:,sizex-1].fill(maxd)                                   #7 Right
        self.diffs[6] = np.logical_or(self.diffs[4], self.diffs[6])*allfill  #6 BottomRight
        self.diffs[8] = np.logical_or(self.diffs[6], self.diffs[0])*allfill  #8 RightTop, 0 Center is full of zeroes
        self.dirs = np.array([[ 0,  0],    #0 Center
                              [-1,  0],    #1 Top
                              [-1, -1],    #2 TopLeft
                              [ 0, -1],    #3 Left
                              [ 1, -1],    #4 LeftBottom
                              [ 1,  0],    #5 Bottom
                              [ 1,  1],    #6 BottomRight
                              [ 0,  1],    #7 Right
                              [-1,  1]],   #8 RightTop
                              dtype=np.int16)
        self.crds = np.indices((sizey,sizex,)).transpose(1,2,0)

    def compare(self, imgl, imgr, offsets):
        diffs = np.copy(self.diffs)#.reshape(self.diffs.shape+[2, 2])
        emptyright = np.copy(self.emptyright)
        emptyright[1:self.szy+1, 1:self.szx+1] = imgr
        dirs = self.dirs+np.array([1, 1], dtype=np.int16)
        for i in range(0, len(dirs)):
            #dy, dx = dirs[i]                        #Directions to shift (normalized)
            #ey, ex = [self.szy+dy, self.szx+dx]     #Ends
            #for gy in range(0, self.szy):
            #    for gx in range(0, self.szx):
            #        ox, oy = offsets[gy, gx] #Previvious offsets (scaled)
            #        diffs[i, gy, gx] += abs(emptyright[(dy+gy+oy),(dx+gx+ox)] - imgl[gy, gx]) #Needs changes

            cmatr = (self.crds+dirs[i]+offsets)\
                    .astype(np.int32)\
                    .transpose(2,0,1) #Transpose back
            diffs[i] += abs(emptyright[cmatr[0], cmatr[1]] - imgl)

        sm = diffs.argmin(axis=0)          #To find best offset indexes
        offs = self.dirs[sm]+offsets   #Find new offsets and append the olds
        print "---Offsets indices==\n", sm
        ###To scale by nearest###
        #s2[::2,::2,:] = s 
        #s2[1::2] = s2[::2] 
        #s2[:,1::2,:] = s2[:,::2,:] 
        print "------Diffs==\n", diffs
        return offs

def vis_offsets(im0, im1, offsets):
    scale = 8
    y, x = im0.shape
    newy, newx = y*scale, x*scale
    crdsfrom = np.indices(im0.shape).transpose(1,2,0) 
    crdsto = (crdsfrom + offsets).astype(np.uint32) #+ np.array([0, x])
    crdsfrom *= scale
    crdsto *= scale
    crdsfrom += scale>>1
    crdsto += scale>>1
    cv_img0rs = cv2.resize(im0, (newx, newy), interpolation = cv2.INTER_NEAREST)
    cv_img1rs = cv2.resize(im1, (newx, newy), interpolation = cv2.INTER_NEAREST) 

    vis = np.zeros((newy, 2*newx), cv_img0.dtype);
    vis[:newy, :newx] = cv_img0rs
    vis[:newy, newx:2*newx] = cv_img1rs
    #print "Coordinates from is"
    #print crdsfrom
    #print "Coordinates to is"
    #print crdsto
    #print "Offstes is"
    #print offsets
    pil_im = Image.fromarray(vis)
    draw = ImageDraw.Draw(pil_im)
    for y in range(0, offsets.shape[0]):
        for x in range(0, offsets.shape[1]):
            if sum(abs(offsets[y, x])) > 0 and randint(0, y)==0:
                coords = tuple(np.concatenate((crdsfrom[y,x][::-1], crdsto[y,x][::-1],)))
                print "Draw to", coords 
                draw.line(coords, fill=255, width=1)
                ex, ey = coords[2:]
                draw.ellipse((ex-1, ey-1, ex+1, ey+1))
    pil_im.show()



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

y = 16
x = 16

comparators = []
offs = np.zeros((y,x,2)).astype(np.int16)

while x<oldx and y<oldy:
    x*=2
    y*=2
    offsets = (offs*2).repeat(2,axis=0).repeat(2,axis=1)
    comparators.append(comparator(y, x, cv_img0.dtype))
    szstr = '_'+str(x)+'x'+str(y)
    cv_img0rs = cv2.resize(cv_img0, (x, y), interpolation = cv2.INTER_AREA)
    cv_img1rs = cv2.resize(cv_img1, (x, y), interpolation = cv2.INTER_AREA) 
    #cv_img0rs = cv2.resize(cv_img0rs, (newx, newy), interpolation = cv2.INTER_NEAREST)
    #cv_img1rs = cv2.resize(cv_img1rs, (newx, newy), interpolation = cv2.INTER_NEAREST) 

    #vis = np.zeros((newy, 2*newx), cv_img0.dtype);
    #vis[:newy, :newx] = cv_img0rs
    #vis[:newy, newx:2*newx] = cv_img1rs
    #cv2.imwrite(dir2+'both'+szstr+ext, vis)

    offs = comparators[-1].compare(cv_img0rs, cv_img1rs, offsets)
    print "----Offsets is\n", offs
    if y>32:
        vis_offsets(cv_img0rs, cv_img1rs, offs)
        exit()
