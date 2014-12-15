import numpy as np
import cv2
from time import clock
from PIL import Image, ImageDraw
from math import log

class comparator():
    def __init__(self, sizey, sizex, dtype):
        self.emptyright = np.zeros((sizey+2, sizex+2), np.int16)
        self.emptyleft  = np.zeros((sizey+2, sizex+2), np.int16)
        self.szx = sizex
        self.szy = sizey
        maxd = np.iinfo(dtype).max*4
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
        ex = np.indices((sizey, sizex,)).astype(np.uint)
        er = np.empty([sizey, sizex, 2]).astype(np.uint)
        er[:,:,1] = ex[1]
        er[:,:,0] = ex[0]
        self.crds = er.astype(np.uint)

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
                    diffs[i, gy, gx] += abs(emptyright[(dy+gy+oy),(dx+gx+ox)] - imgl[gy, gx]) #Needs changes
            #cmatr = (self.crds+dirs[i]+offsets).astype(np.uint)
            #diffs[i] += abs(emptyright[cmatr] - imgl)
        sm = diffs.argmin(axis=0)          #To find best offset indexes
        offs = self.dirs[sm]+offsets   #Find new offsets and append the olds
        ###To scale by nearest###
        #s2[::2,::2,:] = s 
        #s2[1::2] = s2[::2] 
        #s2[:,1::2,:] = s2[:,::2,:] 
        print "------Diffs==\n", diffs
        return offs

def vis_offsets(im0, im1, offsets):
    scale = 16
    y, x = im0.shape
    newy, newx = y*scale, x*scale
    crdsfrom = np.indices(im0.shape).transpose(1,2,0)
    crdsto = (crdsfrom + np.array([0, x]) + offsets).astype(np.uint32)
    crdsfrom *= scale
    crdsto *= scale
    crdsfrom += scale>>1
    crdsto += scale>>1
    cv_img0rs = cv2.resize(im0, (newx, newy), interpolation = cv2.INTER_NEAREST)
    cv_img1rs = cv2.resize(im1, (newx, newy), interpolation = cv2.INTER_NEAREST) 

    vis = np.zeros((newy, 2*newx), cv_img0.dtype);
    vis[:newy, :newx] = cv_img0rs
    vis[:newy, newx:2*newx] = cv_img1rs
    #cv2.imwrite(dir2+'both'+szstr+ext, vis)

    pil_im = Image.fromarray(vis)
    draw = ImageDraw.Draw(pil_im)
    for y in range(0, offsets.shape[0]):
        for x in range(0, offsets.shape[1]):
            if sum(abs(offsets[y, x])) > 0:
                coords = tuple(np.concatenate((crdsfrom[y,x][::-1], crdsto[y,x][::-1],)))
                print "Draw to", coords 
                draw.line(coords, fill=255, width=1)
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
    cv_img1rs = cv2.resize(cv_img1, (x, y), interpolation = cv2.INTER_AREA) 
    #cv_img0rs = cv2.resize(cv_img0rs, (newx, newy), interpolation = cv2.INTER_NEAREST)
    #cv_img1rs = cv2.resize(cv_img1rs, (newx, newy), interpolation = cv2.INTER_NEAREST) 

    #vis = np.zeros((newy, 2*newx), cv_img0.dtype);
    #vis[:newy, :newx] = cv_img0rs
    #vis[:newy, newx:2*newx] = cv_img1rs
    #cv2.imwrite(dir2+'both'+szstr+ext, vis)

    offs = comparators[-1].compare(cv_img0rs, cv_img0rs, offsets)
    print "----Offsets is\n", offs
    if y>4:
        vis_offsets(cv_img0rs, cv_img1rs, offs)
        exit()
    offsets = (offs*2).repeat(2,axis=0).repeat(2,axis=1)
