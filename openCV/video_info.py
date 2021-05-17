#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
import cv2 as cv
from FINAL_IMG_PREP import *


# In[1]:


def vid_info(path, codec, name):
    cap = cv2.VideoCapture(path)
    print(cap)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    #재생할 파일의 높이 얻기
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    #재생할 파일의 프레임 레이트 얻기
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    #codec
    fourcc = cv2.VideoWriter_fourcc(*codec)
    
    #filename
    filename = name+'.mp4'
    
    #out 
    out = cv2.VideoWriter(filename, fourcc, fps, (int(width), int(height)))
    
    print('cap {0}, width {1}, height {2}, fps {3}'.format(cap, width, height, fps))
    print('codec {0}', 'filename {1}'.format(fourcc, filename))
    
    return cap, width, height, fps, fourcc, filename, out


# In[2]:


def background(cap):
    ret, back = cap.read()
    if not ret:
        print('Image registration failure')
        
    back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
    back = cv2.GaussianBlur(back, (0,0),3)
    return back


# In[ ]:

def frame_extractor(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    success, image = cap.read()

    count=0

    while success :
        cv2.imwrite(save_path+'/%d.jpg' % count, image)
        success, image = cap.read()
        print("saved image %d.jpg" % count)
        count +=1


