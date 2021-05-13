#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import glob
import math
get_ipython().run_line_magic('matplotlib', 'inline')

#cv
import cv2
import math
from PIL import Image
import math
from scipy import ndimage
import argparse
import imutils


#시각화
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib import font_manager, rc
rc('font',family="AppleGothic")
plt.rcParams["font.family"]="AppleGothic" #plt 한글꺠짐
plt.rcParams["font.family"]="Arial" #외국어꺠짐
plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 출력 설정
plt.rc('figure', figsize=(10,8))

sns.set(font="AppleGothic", 
        rc={"axes.unicode_minus":False},
        style='darkgrid') #sns 한글깨짐


# In[3]:


def direct_show(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    plt.figure(figsize = (10,8))
    #xticks/yticks - 눈금표
    plt.xticks([])
    plt.yticks([])
    #코랩에서 안돌아감 주의
    plt.imshow(img, cmap= 'gray')
    plt.show()


# In[6]:


def INPUT_IMG(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


# In[2]:


def show(img):
    #사이즈
    plt.figure(figsize = (10,8))
    #xticks/yticks - 눈금표
    plt.xticks([])
    plt.yticks([])
    #코랩에서 안돌아감 주의
    plt.imshow(img, cmap= 'gray')
    plt.show()


# In[3]:


#이미지 수 확인하기
def count_img(path):
    data_path = os.path.join(path, '*g')
    files= glob.glob(data_path)
    img_list=[]
    for f1 in files:
        img = cv2.imread(f1)
        img_list.append(img)
    print('이미지수',len(img_list)) 


# In[4]:


#이미지 불러오기
def get_img(path):
    data_path = os.path.join(path, '*g')
    files= glob.glob(data_path)
    img_list=[]
    for f1 in files:
        img = cv2.imread(f1)
        img_list.append(img)
#     print('이미지수',len(img_list))
#     print('show(get_img(list_file[1])[0]) 식으로 이미지 불러와서 img로 저장')
    
    return img_list
    
# data_img = get_img(list_file[2])
# show(img_list[1])


# In[5]:


#masking, return 까먹지 말기 흑흑 
def get_mask(img):
    # #마스크 생성을 위해, 밝기 강조한 Lab으로 이미지 변환 01
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    # #블러 02
    # #블러의 커널 사이즈가 홀수만 가능하므로 이미지 평균 값을 기준으로 홀수값 만들기
    blur_k = int((img.mean()*0.5)//2)*2+1 
    img = cv2.medianBlur(img, blur_k)
    # #threshold 적용을 위해 Lab에서 Grayscale로 이미지 변환 03
    img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # #이미지 평균값을 기준으로 이진화 04
    ret, img = cv2.threshold(img, img.mean()*1.1, 255, cv2.THRESH_BINARY)

    # # #가장 큰 값의 컨투어로 마스크 만들기 05
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_cnt = max(contours, key=cv2.contourArea)
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.drawContours(mask, [max_cnt], -1, (255,255,255), -1)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.dilate(mask,k)
    return mask

#img_cropping 
def get_cropped_mask(img, mask):
    """
    마스크를 기준으로 경계선을 찾아 위/왼/오른쪽을 자루는 함수로서
    img = original image
    mask = bit_img
    cropped_img = 원본 이미지에서 마크된 영역을 갖는 부분 반환
    """
    
    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img_.shape
    
    #마스크 기준으로 위/왼/오른쪽 경계선 찾기(숫자로 확인 가능)
    mask_list = mask.tolist()
    
    #테두리가 흰색인 경우를 고려해서, 테두리에서 5% 지점부터 경계점 찾기 시작
    #경계점은 중간 부분(30~70%)에서 검은색(0)을 벗어난 지점을 기준으로 함
    #위쪽
    for y in range(int(height*0.05), height): #마스크이미지에서, 일반 이미지의 5%이상의 지점에서 
    #가로는 30%-70%까지가 0보다 클때 (마스크의 max값이 - 그 범위에 1(흰색)이 있을때)
        if max(mask[y,int(width*0.3):int(width*0.7)]) >0:
        #총 mask 이미지에서, 일반이미지에서 5%더한 값을 뺌
            start_y = y-int(height*0.05)
            break
    
    #왼쪽 start point
    for x in range(int(width*0.05),width):
        if max(mask[x,int(height*0.3):int(height*0.7)]) >0:
            start_x = x-int(width*0.05)
            break

    # #오른쪽, stop, -1,-1(오른쪽에서 왼쪽으로)
    for x in range(int(width*0.95),-1,-1):
        if max(mask[int(height*0.3):int(height*0.7),x]) > 0:
            end_x = x+int(width*0.05)
            break

    #경계선 기준으로 이미지와 마스크 자름
    img_ = img_[start_y:,start_x:end_x]
    mask = mask[start_y:,start_x:end_x]

    img = cv2.bitwise_and(img_, mask)
    
    return img

def wrist_cut(img):
    height = img.shape[0]
    width = img.shape[1]

    #이미지의 아래에서부터 시작해서 화소 평균이 커지는(밝아지는) 경계선 찾기
    start = int(height*0.95)  #아래 테두리가 밝은 경우를 고려해서 height*0.95부터 시작함
    index = 0
    k = 10 #10개 행씩 평균 구함
    while True:
        pixel_lower = img[start-k*(index+1):start-k*index,:].mean()
        pixel_upper = img[start-k*(index+2):start-k*(index+1),:].mean()
        if pixel_upper - pixel_lower > 0:
            end_y = start-k*(index+1)
            break
        index += 1

    img = img[:end_y]
    return img

def mask_for_center(img):
    blur_k = int((img.mean()*0.5)//2)*2+1 
    img = cv2.medianBlur(img, blur_k)

    # #이미지 평균값을 기준으로 이진화 04
    ret, img = cv2.threshold(img, img.mean()*1.1, 255, cv2.THRESH_BINARY)

    # # #가장 큰 값의 컨투어로 마스크 만들기 05
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_cnt = max(contours, key=cv2.contourArea)
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.drawContours(mask, [max_cnt], -1, (255,255,255), -1)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.dilate(mask,k)
    return mask



#img_preprocessing
def blake_back(img):
    mask =get_mask(img)
    black_back = get_cropped_mask(img, mask)
    black_back = wrist_cut(black_back)
    center_mask = mask_for_center(black_back)
    return black_back, center_mask


# In[1]:


def get_center(center_mask):
#     mask = mask_for_center(img)
    res, thresh = cv2.threshold(center_mask, center_mask.mean(), 255, cv2.THRESH_BINARY)

    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    
    return cX, cY

def get_far_list(img):
    #손가락포인트
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(max_cnt, returnPoints = False)
    defects = cv2.convexityDefects(max_cnt, hull)
    
    far_list = []
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(max_cnt[s][0])
        end = tuple(max_cnt[e][0])
        far = tuple(max_cnt[f][0])
        far_list.append(far)
        
    return far_list

def get_pinky_point(far_list):
    far_list.sort(key = lambda x:x[0])
    pX, pY = far_list[0]
    return pX, pY

def get_thumbs_point(far_list):
    far_list.sort(key = lambda x:x[0])
    thX, thY = far_list[-1]
    return thX, thY

def get_middle_point(far_list):
    far_list.sort(key = lambda x:x[1])
    tX, tY = far_list[0]
    return tX, tY

#point
def get_pinky_finger_point(center_mask, black_back):
    cX, cY = get_center(center_mask)
    far_list = get_far_list(black_back)
    pX, pY = get_pinky_point(far_list)
    return cX, cY, pX, pY

def get_thumbs_finger_point(center_mask, black_back):
    cX, cY = get_center(center_mask)
    far_list = get_far_list(black_back)
    thX, thY = get_thumbs_point(far_list)
    return cX, cY, thX, thY

def get_middle_finger_point(center_mask, black_back):
    cX, cY = get_center(center_mask)
    far_list = get_far_list(black_back)
    tX, tY = get_middle_point(far_list)
    return cX, cY, tX, tY


# In[7]:


def rotation_cut(img):
    img = ndimage.rotate(img, 70)
    ret, th = cv2.threshold(img, img.mean(), 255, cv2.THRESH_BINARY)
    th_l = th.tolist()
    cut_index = 0
    if th_l[0][0] == 0 or th_l[0][-1] == 0:
        for i in reversed(range(len(th_l))):
            if th_l[i].count(255) > 0:
                cut_index = i

    img = img[cut_index:]
    return img

def center_img(img):
    imgY, imgX = img.shape[:2]
    imgY = int((imgY)/2)
    imgX = int(imgX/2)
    
    return imgY, imgX


# In[8]:


def pinky_rotation(img):
    angle = math.degrees(math.atan2(cY-thY, cX-thX))
    img = ndimage.rotate(img, angle-100)#시계방향
    return img

def rotation_cut(img):
    ret, th = cv2.threshold(img, img.mean(), 255, cv2.THRESH_BINARY)
    th_l = th.tolist()
    cut_index = 0
    if th_l[0][0] == 0 or th_l[0][-1] == 0:
        for i in reversed(range(len(th_l))):
            if th_l[i].count(255) > 0:
                cut_index = i

    img = img[cut_index:]
    return img

def center_img(img):
    imgY, imgX = img.shape[:2]
    imgY = int((imgY)/2)
    imgX = int(imgX/2)
    
    return imgY, imgX


#rotation / get_imgYX
def get_imgYX(img):
    rotated_img = pinky_rotation(img)
    show(rotated_img)
    img = rotation_cut(rotated_img)
    show(img)
    imgY, imgX = center_img(img)
    return img, imgY, imgX


# In[1]:


def get_img_croped(img):
    ret, thresh = cv2.threshold(img, img.mean(), 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_xy = np.array(contours)

    # x의 min과 max 찾기
    x_min, x_max = 0,0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][0]) #네번째 괄호가 0일때 x의 값
            x_min = min(value)
            x_max = max(value)
#     print(x_min)
#     print(x_max)

    # y의 min과 max 찾기
    y_min, y_max = 0,0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][1]) #네번째 괄호가 0일때 x의 값
            y_min = min(value)
            y_max = max(value)
#     print(y_min)
#     print(y_max)

    # image trim 하기
    x = x_min
    y = y_min
    w = x_max-x_min
    h = y_max-y_min

    img_trim = img[y:y+h, x:x+w]
    
    return img_trim 


# In[ ]:





# In[8]:


#get skin removed
def binarization3(path):
#     img = cv2.imread(path, cv2.IMREAD_COLOR)
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hist = cv2.equalizeHist(img_gray)
    ret, thr = cv2.threshold(img_hist,img_gray.mean(),255,cv2.THRESH_TOZERO)
    img_hist = cv2.equalizeHist(thr)
    ret1, thr1 = cv2.threshold(img_hist,thr.mean(),255,cv2.THRESH_TOZERO)
    clahe = cv2.createCLAHE(clipLimit=img_gray.mean(), tileGridSize=(1,1))
    clahe_img = clahe.apply(thr1)
    final_img = cv2.threshold(clahe_img, clahe_img.mean(),255, cv2.THRESH_TOZERO)[1]
    return final_img


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#WRIST_ROI

import glob
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
import seaborn as snsx
import math
import glob

#cv
from PIL import Image
import math
from scipy import ndimage
import argparse
import imutils



#시각화
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from matplotlib import font_manager, rc
rc('font',family="AppleGothic")
plt.rcParams["font.family"]="AppleGothic" #plt 한글꺠짐
plt.rcParams["font.family"]="Arial" #외국어꺠짐
plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 출력 설정
plt.rc('figure', figsize=(10,8))

sns.set(font="AppleGothic", 
        rc={"axes.unicode_minus":False},
        style='darkgrid') #sns 한글깨짐


#이미지 수 확인하기
def count_img(path):
    data_path = os.path.join(path, '*g')
    files= glob.glob(data_path)
    img_list=[]
    for f1 in files:
        img = cv2.imread(f1)
        img_list.append(img)
    print('이미지수',len(img_list))

#이미지 불러오기
def get_img(path):
    data_path = os.path.join(path, '*g')
    files= glob.glob(data_path)
    img_list=[]
    for f1 in files:
        img = cv2.imread(f1)
        img_list.append(img)
#     print('이미지수',len(img_list))
#     print('show(get_img(list_file[1])[0]) 식으로 이미지 불러와서 img로 저장')
    
    return img_list
    
# data_img = get_img(list_file[2])
# show(img_list[1])

#이미지 함수
def show(img):
    #사이즈
    plt.figure(figsize = (10,8))
    #xticks/yticks - 눈금표
    plt.xticks([])
    plt.yticks([])
    #코랩에서 안돌아감 주의
    plt.imshow(img, cmap= 'gray')
    plt.show()

#masking, return 까먹지 말기 흑흑 
def get_mask(img):
#     img = cv2.imread(img_path+"/org001.jpg")
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # #마스크 생성을 위해, 밝기 강조한 Lab으로 이미지 변환 01
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    # show(img)

    # #블러 02
    # #블러의 커널 사이즈가 홀수만 가능하므로 이미지 평균 값을 기준으로 홀수값 만들기
    blur_k = int((img.mean()*0.5)//2)*2+1 
    img = cv2.medianBlur(img, blur_k)
    # show(img)
    # #threshold 적용을 위해 Lab에서 Grayscale로 이미지 변환 03
    img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # #이미지 평균값을 기준으로 이진화 04
    ret, img = cv2.threshold(img, img.mean()*1.1, 255, cv2.THRESH_BINARY)

    # # #가장 큰 값의 컨투어로 마스크 만들기 05
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_cnt = max(contours, key=cv2.contourArea)
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.drawContours(mask, [max_cnt], -1, (255,255,255), -1)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.dilate(mask,k)
    return mask

#     show(mask)
    #그 다음에 가지고 있는 이미지 모두 마스크 처리 --> 오래걸림
def binarization3(img):
    img_c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_gray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    img_hist = cv2.equalizeHist(img_gray)
    ret, thr = cv2.threshold(img_hist,img_gray.mean(),255,cv2.THRESH_TOZERO)
    img_hist = cv2.equalizeHist(thr)
    ret1, thr1 = cv2.threshold(img_hist,thr.mean(),255,cv2.THRESH_TOZERO)
    clahe = cv2.createCLAHE(clipLimit=img_gray.mean(), tileGridSize=(1,1))
    clahe_img = clahe.apply(thr1)
    final_img = cv2.threshold(clahe_img, clahe_img.mean(),255, cv2.THRESH_TOZERO)[1]
    return final_img


def mask_for_center(img):
    blur_k = int((img.mean()*0.5)//2)*2+1 
    img = cv2.medianBlur(img, blur_k)
    
    # #이미지 평균값을 기준으로 이진화 04
    ret, img = cv2.threshold(img, img.mean()*1.1, 255, cv2.THRESH_BINARY)

    # # #가장 큰 값의 컨투어로 마스크 만들기 05
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_cnt = max(contours, key=cv2.contourArea)
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.drawContours(mask, [max_cnt], -1, (255,255,255), -1)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.dilate(mask,k)
    return mask

#img_cropping 
def get_cropped_mask(img, mask):
    """
    마스크를 기준으로 경계선을 찾아 위/왼/오른쪽을 자루는 함수로서
    img = original image
    mask = bit_img
    cropped_img = 원본 이미지에서 마크된 영역을 갖는 부분 반환
    """
    
    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img_.shape
    
    #마스크 기준으로 위/왼/오른쪽 경계선 찾기(숫자로 확인 가능)
    mask_list = mask.tolist()
    
    #테두리가 흰색인 경우를 고려해서, 테두리에서 5% 지점부터 경계점 찾기 시작
    #경계점은 중간 부분(30~70%)에서 검은색(0)을 벗어난 지점을 기준으로 함
    #위쪽
    for y in range(int(height*0.05), height): #마스크이미지에서, 일반 이미지의 5%이상의 지점에서 
    #가로는 30%-70%까지가 0보다 클때 (마스크의 max값이 - 그 범위에 1(흰색)이 있을때)
        if max(mask[y,int(width*0.3):int(width*0.7)]) >0:
        #총 mask 이미지에서, 일반이미지에서 5%더한 값을 뺌
            start_y = y-int(height*0.05)
            break
    
    #왼쪽 start point
    for x in range(int(width*0.05),width):
        if max(mask[x,int(height*0.3):int(height*0.7)]) >0:
            start_x = x-int(width*0.05)
            break

    # #오른쪽, stop, -1,-1(오른쪽에서 왼쪽으로)
    for x in range(int(width*0.95),-1,-1):
        if max(mask[int(height*0.3):int(height*0.7),x]) > 0:
            end_x = x+int(width*0.05)
            break

    #경계선 기준으로 이미지와 마스크 자름
    img_ = img_[start_y:,start_x:end_x]
    mask = mask[start_y:,start_x:end_x]

    cropped_mask_img = cv2.bitwise_and(img_, mask)
    
    return cropped_mask_img

def wrist_cut(img):
    height = img.shape[0]
    width = img.shape[1]

    #이미지의 아래에서부터 시작해서 화소 평균이 커지는(밝아지는) 경계선 찾기
    start = int(height*0.95)  #아래 테두리가 밝은 경우를 고려해서 height*0.95부터 시작함
    index = 0
    k = 10 #10개 행씩 평균 구함
    while True:
        pixel_lower = img[start-k*(index+1):start-k*index,:].mean()
        pixel_upper = img[start-k*(index+2):start-k*(index+1),:].mean()
        if pixel_upper - pixel_lower > 0:
            end_y = start-k*(index+1)
            break
        index += 1

    img = img[:end_y]
    return img


def get_center(img):
    mask = mask_for_center(img)
    res, thresh = cv2.threshold(mask, mask.mean(), 255, cv2.THRESH_BINARY)

    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    center = cX, cY
    return center

def get_finger_point(img):
    #손가락포인트
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(max_cnt, returnPoints = False)
    defects = cv2.convexityDefects(max_cnt, hull)

    start_list = []
    far_list = []
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(max_cnt[s][0])
        end = tuple(max_cnt[e][0])
        far = tuple(max_cnt[f][0])
        far_list.append(far)

        ##중지 
        for i in far_list:
            far_list.sort(key= lambda x:x[1])
            top = far_list[0]
            tX, tY = top
#             mask = mask_for_center(img)
            cX, cY = get_center(img)
#             cv2.circle(img, top, 5, (2, 255, 180),10)
            
    #새끼 손가락        
    far_list.sort(key= lambda x:x[0])
    pinky_list = []
    for i in far_list:
        if i[1] < cY and i[0] < cX:
            pinky_list.append(i)
    pinky_list.sort(key= lambda x:x[0])
    pinky = pinky_list[0]
    pX, pY = pinky
#     cv2.circle(img, pinky, 5, (2, 255, 180),10)

    # 엄지손가락
    far_list.sort(key= lambda x:x[0])
    thumbs = far_list[-1]
    thX, thY = thumbs
#     cv2.circle(img, thumbs, 5, (2, 255, 180),10)
    
    return top, pinky, thumbs


#중지기준 rotation
def middle_rotation(img):
    cX, cY = get_center(img)
    top, pinky, thumbs = get_finger_point(img)
    tX, tY = top
    angle = math.degrees(math.atan2(cY-tY, cX-tX))
    img = ndimage.rotate(img, angle-90)#시계방향
    return img

#엄지기준 rotation
def thumbs_rotation(img):
    cX, cY = get_center(img)
    top, pinky, thumbs = get_finger_point(img)
    thX, thY = thumbs
    angle = math.degrees(math.atan2(cY-thY, cX-thX))
    img = ndimage.rotate(img, angle-90)#시계방향
    return img

#새끼손가락 기준 rotation
def pinky_rotation(img):
    cX, cY = get_center(img)
    top, pinky, thumbs = get_finger_point(img)
    pX, pY = pinky
    angle = math.degrees(math.atan2(cY-pY, cX-pX))
    img = ndimage.rotate(img, angle-90)#시계방향
    return img

def rotation_cut(img):
    ret, th = cv2.threshold(img, img.mean(), 255, cv2.THRESH_BINARY)
    th_l = th.tolist()
    cut_index = 0
    if th_l[0][0] == 0 or th_l[0][-1] == 0:
        for i in reversed(range(len(th_l))):
            if th_l[i].count(255) > 0:
                cut_index = i

    img = img[cut_index:]
    return img

def center_img(img):
    imgY, imgX = img.shape[:2]
    imgY = int((imgY)/2)
    imgX = int(imgX/2)
    
    return imgY, imgX

#중지자르기
def middle_cut(img):
    top, pinky, thumbs = get_finger_point(img)
    tX, tY = top
    imgY, imgX = center_img(img)
    middle_cut =img[0:imgY, tX-150:tX+200]
    middle_cut = cv2.resize(middle_cut,(100,500), cv2.INTER_AREA)
    return middle_cut

#엄지자르기
#img[y:y, x:x]
def thumbs_cut(img):
    imgY, imgX = center_img(img)
    thumbs_cut =img[0:imgY-70, imgX:imgX+1300]
    thumbs_cut = cv2.rotate(thumbs_cut, cv2.ROTATE_90_CLOCKWISE)
    thumbs_cut = cv2.resize(thumbs_cut,(100,500), cv2.INTER_AREA)
    return thumbs_cut

#새끼손가락
def pinky_cut(img):
    imgY, imgX = center_img(img)
    pinky_cut =img[0:imgY-100, imgX-400:imgX]
    pinky_cut = cv2.resize(pinky_cut,(100,500), cv2.INTER_AREA)
    return pinky_cut

def get_thumbs_finger(img):
    
    #마스크처리
    mask = get_mask(img)
    
    #마스크에 맞춰 손윤곽 자르기 여기서 grayscale로 변환됨
    img = get_cropped_mask(img, mask)

    #손목 위 자르기
    img = wrist_cut(img)
    mask = mask_for_center(img)

#     #센터점 구하기
#     center = get_center(mask)

    #손가락포인트구하기
#     top, pinky, thumbs = get_finger_point(img)

    #엄지기준 rotation
    img = thumbs_rotation(img)

    #상위에 맞춰 자르기
    img = rotation_cut(img)

#     #이미지 중앙값
#     imgY, imgX = center_img(img)
    
    #thumbs_cut + resize
    thumbs_cut_img = thumbs_cut(img)
    thumbs_cut_img = binarization3(thumbs_cut_img)
    thumbs_cut_img = cv2.resize(thumbs_cut_img, (100,500),cv2.INTER_AREA)
    return thumbs_cut_img

def get_middle_finger(img):
    #마스크처리
    mask = get_mask(img)
    
    #마스크에 맞춰 손윤곽 자르기 여기서 grayscale로 변환됨
    img = get_cropped_mask(img, mask)

    #손목 위 자르기
    img = wrist_cut(img)
    mask = mask_for_center(img)

#     #센터점 구하기
#     center = get_center(mask)

#     #손가락포인트구하기
#     top, pinky, thumbs = get_finger_point(img)

    #중지기준 rotation
    img = middle_rotation(img)

    #상위에 맞춰 자르기
    img = rotation_cut(img)

#     #이미지 중앙값
#     imgY, imgX = center_img(img)
    
    #middle_cut + resize
    middle_cut_img = middle_cut(img)
    middle_cut_img = binarization3(middle_cut_img)
    middle_cut_img = cv2.resize(middle_cut_img, (100,500),cv2.INTER_AREA)
    return middle_cut_img



def get_pinky_finger(img):
    
    #마스크처리
    mask = get_mask(img)
    
    #마스크에 맞춰 손윤곽 자르기 여기서 grayscale로 변환됨
    img = get_cropped_mask(img, mask)

    #손목 위 자르기
    img = wrist_cut(img)
    mask = mask_for_center(img)

    #엄지기준 rotation
    img = pinky_rotation(img)

    #상위에 맞춰 자르기
    img = rotation_cut(img)

    #thumbs_cut + resize
    pinky_cut_img = pinky_cut(img)
    pinky_cut_img = binarization3(pinky_cut_img)
    pinky_cut_img = cv2.resize(pinky_cut_img, (100,500),cv2.INTER_AREA)
    return pinky_cut_img


###손목#####
def cut(img):
    """
    마스크를 기준으로 경계선을 찾아 위/왼쪽/오른쪽을 자르는 함수
    Parameters
    img : 원본 이미지 객체
    mask : 마스크된 이미지 객체
    """
    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img_.shape
    mask = get_mask(img)
    #마스크 기준으로 위/왼/오른쪽 경계선 찾기
    mask_list = mask.tolist()
    

    #테두리가 흰색인 경우를 고려해서, 테두리에서 5% 지점부터 경계점 찾기 시작
    #경계점은 중간 부분(30~70%)에서 검은색(0)을 벗어난 지점을 기준으로 함
    #위쪽
    for y in range(int(height*0.05),height):
        if max(mask[y,int(width*0.3):int(width*0.7)]) > 0:
            start_y = y-int(height*0.05)
            break
    #왼쪽
    for x in range(int(width*0.05),width):
        if max(mask[int(height*0.3):int(height*0.7),x]) > 0:
            start_x = x-int(width*0.05)
            break
    #오른쪽
    for x in range(int(width*0.93),-1,-1):
        if max(mask[int(height*0.3):int(height*0.7),x]) > 0:
            end_x = x+int(width*0.05)
            break
            
    #아래쪽
    cut_index = 0
    # 맨 밑 처음이나 끝에 흰색이 나오면 검은색이 나오는 부분까지 자르기
    if mask_list[height-1][-1] == 255 or mask_list[height-1][0] == 255:
        for n in reversed(range(height)):
            if mask_list[n][0] == 0 or mask_list[n][-1] == 0:
                cut_index = n
                break

    # 맨 밑 처음이 검정색이면 흰색이 나오는 부분까지 자르기
    if mask_list[height-1][0] == 0 or mask_list[height-1][-1] == 0:
        for n in reversed(range(height)):
            if mask_list[n].count(255) > 100:
                cut_index = n
                break

    if cut_index == 0:
        cut_index = height

    #경계선 기준으로 이미지와 마스크 자름
    img_ = img_[start_y:(cut_index-1),start_x:end_x]
    mask = mask[start_y:(cut_index-1),start_x:end_x]

    masked = cv2.bitwise_and(img_, mask)
    masked = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
    
    return masked

def rotation(img):
    """
    마스크 기준으로 아래쪽 자르기 및 회전하는 함수
    
    Parameters
    img : 마스크된 이미지 객체
    """
    img = cut(img)
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    th_l = th.tolist()


    # 밑에서부터 처음으로 검은 색이 나오는 부분이 lower
    for n in reversed(range(len(th_l))):
        if th_l[n][0] == 0 and th_l[n][-1] == 0:
            lower = n
            break

    # 만약 맨 밑이 lower면 이미지의 90퍼센트 부분을 lower로 정의
    if lower == h - 1:
        lower = int(len(th_l)*0.9)

    # upper는 lower에서 5퍼센트만큼 올라간 부분
    slice5 = int(len(th)*0.05)
    upper = lower - slice5

    # x, y좌표들은 이미지의 85퍼센트(upper)와 90퍼센트(lower) 부분의 손목 가운데 지점들
    x,y = [],[]
    for n in range(slice5):
        cnt = th_l[n + upper].count(255)
        index = th_l[n + upper].index(255)
        x.append([n+upper])
        y.append([int((index*2 + cnt - 1)/2)])

    # x,y좌표로 단순선형회귀 그리기
    model = LinearRegression()
    model.fit(X=x,y=y)

    # 회전
    angle = math.atan2(h - 0, int(model.predict([[h]])) - int(model.predict([[0]])))*180/math.pi
    M = cv2.getRotationMatrix2D((w/2,h/2),angle-90,1)
    rotate = cv2.warpAffine(img, M, (w,h))

    # 회전한 부분을 자르기
    for n in range(len(th[-1])):
        if th[-1][n] == 255:
            start_x = n
            break

    for n in range(len(th[-1])):
        if th[-1][n] == 255:
            end_x = n

    s_point = h - int((int(model.predict([[h]])-start_x)) * math.tan(math.pi*((90-angle)/180)))
    e_point = h - int((end_x - int(model.predict([[h]]))) * math.tan(math.pi*((angle-90)/180)))
    point = min(s_point, e_point)

    img = rotate[:point]
    
    return img

def find_dots(mask):
    """
    손목ROI 구할 때 convexhull함수를 이용하여 손목시작 점 찾는 함수
    
    Parameters
    mask:rotation된 이미지의 mask 좌표
    """
    height, width = mask.shape
    # dots = [ 중앙점 리스트, 왼편 점들 리스트, 오른편 점들 리스트 ]
    dots = [[], [], []]
    #마스크 기준으로 컨투어
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_cnt = max(contours, key=cv2.contourArea)
    
    # 중심점
    M = cv2.moments(max_cnt)
    center = ( int(M['m10'] / M['m00']), int(M['m01'] / M['m00']) )
    dots[0].append(center)
    
    # 왼쪽, 오른쪽 점
    hull = cv2.convexHull(max_cnt, returnPoints = False)
    defects = cv2.convexityDefects(max_cnt, hull)
    for index in range(defects.shape[0]):
        s,e,f,d = defects[index,0]
        far = tuple(max_cnt[f][0])
        
        #오른쪽 점: far 좌표가 중심점보다 오른쪽 아래에 있고, 이미지 테두리보다 안쪽에 있다
        if (far[0] >= center[0]) and (far[1] >= center[1] + 50) and (far[0] <= width*0.95) and (far[1] <= height*0.95):
            dots[2].append(far)
        #왼쪽 점: far 좌표가 중심점보다 왼쪽 아래에 있고, 이미지의 아래 테두리보다 위에 있다
        elif (far[0] <= center[0]) and (far[1] >= center[1] + 50) and (far[0] <= width*0.95) and (far[1] <= height*0.95):
            dots[1].append(far)
    
    return dots

def find_final_dot(img_height,dots):
    """
    find_dots 함수에서 구한 점들 중 
    가까운 거리에 여러 점 찍힌 것을 단순화하는 함수
    
    
    """
    # 가까운 애들 중에서 큰 애만 선별한다
    
    # 왼편 점들 중에서 2개 이상일 때 점 사이 y 거리가 50이 안될 정도로 가까우면 temp에 추가
    # 그래서 y 값이 가장 큰 값만 남기기
    temp = []
    if len(dots[1]) > 1:
        for n in range(len(dots[1])-1):
            if abs(dots[1][n][1] - dots[1][n+1][1]) < 50:
                temp.append(n)
        if len(temp) != 0:
            for m in reversed(range(len(temp))):
                del dots[1][temp[m]]
                
    # 오른편 점들 중에서 2개 이상일 때 점 사이 y 거리가 50이 안될 정도로 가까우면 temp에 추가
    # 그래서 y 값이 가장 큰 값만 남기기
    temp = []
    if len(dots[2]) > 1:
        for n in range(len(dots[2])-1):
            if abs(dots[2][n][1] - dots[2][n+1][1]) < 50:
                temp.append(n)
        if len(temp) != 0:
            for m in reversed(range(len(temp))):
                del dots[2][temp[m]]
          
    # 왼쪽 애들과 오른쪽 애들 사이의 거리가 가까운 애를 선별하고 최종 점 선정
    # 왼쪽 점과 오른쪽 점의 개수에 따라 나누었음
    
    # 왼쪽 점, 오른쪽 점이 1개 이상일 때 왼편과 오른편으로 나누어 서로 간의 거리를 계산함
    # 거리가 가장 짧은 왼편 점과 오른편 점을 선정 -> left, right
    dist_list = []
    if (len(dots[1]) >= 1) and (len(dots[2]) >= 1):
        for m in range(len(dots[1])):
            for n in range(len(dots[2])):
                dist = distance(dots[1][m],dots[2][n])
                dist_list.append([m,n,dist])
        temp = []
        for n in range(len(dist_list)):
            temp.append(dist_list[n][2])
        left = dots[1][dist_list[temp.index(min(temp))][0]]
        right = dots[2][dist_list[temp.index(min(temp))][1]]

        # 중심점에서 왼쪽 오른쪽까지 거리 비율이 40프로 이내인 점을 선정하기 위해
        left_dist = distance(dots[0][0],left)
        right_dist = distance(dots[0][0],right)
        # 왼쪽 점이 중심점에서 지나치게 멀면 오른쪽이 최종점
        if (round(left_dist/img_height,2) > 0.4) and (round(right_dist/img_height,2) <= 0.4):
            final_dot = right
        # 오른쪽 점이 중심점에서 지나치게 멀면 왼쪽이 최종점
        elif (round(left_dist/img_height,2) <= 0.4) and (round(right_dist/img_height,2) > 0.4):
            final_dot = left
        # 둘 다 너무 안 멀면 좀 더 밑에 있는 점이 최종점
        elif (round(left_dist/img_height,2) <= 0.4) and (round(right_dist/img_height,2) <= 0.4):
            if left[1] > right[1]:
                final_dot = left
            else:
                final_dot = right
        # 잘못 나오면 에러 뜨게
        else:
            final_dot = 0
            
    # 왼쪽 점만 1개 이상일 때 가장 y값이 큰 값을 최종점으로 지정
    elif (len(dots[1]) >= 1) and (len(dots[2]) == 0):
        temp = []
        for n in range(len(dots[1])):
            temp.append(dots[1][n][1])
        final_dot = dots[1][temp.index(max(temp))]
        
    # 오른쪽 점만 1개 이상일 때 가장 y값이 큰 값을 최종점으로 지정
    elif (len(dots[1]) == 0) and (len(dots[2]) >= 1):
        temp = []
        for n in range(len(dots[2])):
            temp.append(dots[2][n][1])
        final_dot = dots[2][temp.index(max(temp))]
    
    return final_dot

# 거리 계산
def distance(dot1, dot2):
    dst_x = abs(dot1[0] - dot2[0])
    dst_y = abs(dot1[1] - dot2[1])
    dist = round(math.sqrt(dst_x*dst_x + dst_y*dst_y),2)
    
    return dist

def roi(img):
    """
    손목ROI 구하는 함수
    
    Parameters
    img : rotation된 이미지
    """
    img = rotation(img)
    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
    height, width = img_.shape
    ret, mask = cv2.threshold(img_, 10, 255, cv2.THRESH_BINARY)
    
    # 점들 찾기 함수 적용
    dots = find_dots(mask)
    
    # 이미지에서 뼈 부분 높이 계산
    for row_h in range(len(mask)):
        if mask[row_h].max() == 255:
            img_height = height - row_h
            break

    # 오른쪽 아래에 있는 점이 없는 경우, resize해서 함수 재실행
    if len(dots[2]) == 0:
        resize_height = 800
        resize_width = 600
        img = cv2.resize(img, (resize_height, resize_width), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (resize_height, resize_width), interpolation=cv2.INTER_AREA)
        dots = find_dots(mask)

    # 최종 점 선정
    final_dot = find_final_dot(img_height, dots)
    
    # 최종 점이 왼쪽이면 같은 높이의 오른쪽 점을 찾기
    if abs(final_dot[0] - np.argmax(mask[final_dot[1],:])) < 30:
        for col in reversed(range(len(mask[final_dot[1]]))):
            if mask[final_dot[1]][col] == 255:
                right_x = col
                break
    else:
        right_x = final_dot[0]
    end_y = int(final_dot[1]*1.1)
    left_x = np.argmax(mask[final_dot[1],:])
    start_y = int(final_dot[1]*0.75)
    
    roi = img[start_y:end_y,left_x:right_x]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = binarization3(roi)
    roi = cv2.resize(roi, (224,224),cv2.INTER_AREA)
    
    return roi


# In[2]:


#새끼손가락
def get_pinky_img (path, hY, lY, xx, xy):
    img = INPUT_IMG(path)
    black_back, center_mask = blake_back(img)
    cX, cY, pX, pY = get_pinky_finger_point(center_mask, black_back)
    pinky_test = black_back[pY-hY:cY-lY, pX+xx:pX+xy]
    pinky_fin = get_img_croped(pinky_test)
    final_pinky = binarization3(pinky_fin)
    rgb = cv2.cvtColor(final_pinky, cv2.COLOR_GRAY2BGR)
    return rgb

#엄지손가락
def get_thumbs_img (path, hY, lY, xx, xy):
    img = INPUT_IMG(path)
    black_back, center_mask = blake_back(img)
    cX, cY, thX, thY = get_thumbs_finger_point(center_mask, black_back)
    thumbs_test = black_back[thY+hY:cY+lY, thX+xx:thX+xy]
    thumbs_fin = get_img_croped(thumbs_test)
    final_thumbs= binarization3(thumbs_fin)
    rgb = cv2.cvtColor(final_thumbs, cv2.COLOR_GRAY2BGR)
    return rgb

#가운데손가락
def get_middle_img (path, hY, lY, xx, xy):
    img = INPUT_IMG(path)
    black_back, center_mask = blake_back(img)
    cX, cY, tX, tY = get_middle_finger_point(center_mask, black_back)
    middle_test = black_back[tY+hY:cY+lY, tX+xx:cX+xy]
    middle_fin = get_img_croped(middle_test)
    final_middle= binarization3(middle_fin)
    rgb = cv2.cvtColor(final_middle, cv2.COLOR_GRAY2BGR)
    return rgb


# In[1]:


# model input
def make_input_img(wrist, pinky, thumbs, middle):
    wrist = cv2.cvtColor(wrist, cv2.COLOR_GRAY2BGR)
    wrist = cv2.resize(wrist,(224,224))
    pinky = cv2.resize(pinky,(50,250))
    thumbs = cv2.resize(thumbs,(50,250))
    middle = cv2.resize(middle,(50,250))
    
    return wrist, pinky, thumbs, middle


# In[ ]:




