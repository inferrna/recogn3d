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

while x>2 and y>2:
    x//=2
    y//=2
    szstr = '_'+str(x)+'x'+str(y)
    cv_img0rs = cv2.resize(cv_img0, (x, y), interpolation = cv2.INTER_AREA) 
    cv_img0rs = cv2.resize(cv_img0rs, (newx, newy), interpolation = cv2.INTER_NEAREST) 
    cv_img1rs = cv2.resize(cv_img1, (x, y), interpolation = cv2.INTER_AREA) 
    cv_img1rs = cv2.resize(cv_img1rs, (newx, newy), interpolation = cv2.INTER_NEAREST) 
    #cv2.imwrite(dir2+rnm+szstr+ext, cv_img0rs)
    #cv2.imwrite(dir2+lnm+szstr+ext, cv_img1rs)
    vis = np.zeros((newy, 2*newx), cv_img0.dtype);
    vis[:newy, :newx] = cv_img0rs
    vis[:newy, newx:2*newx] = cv_img1rs
    cv2.imwrite(dir2+'both'+szstr+ext, vis)
