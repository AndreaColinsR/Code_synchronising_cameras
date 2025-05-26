#!/usr/bin/env python
# coding: utf-8

# # Record simultaneous calibration videos while streaming

# In[1]:


import numpy as np
import cv2 as cv
import time
import matplotlib.pyplot as plt


# In[2]:


def sidebyside(videoA,videoB,VideoConcat):
    # open videos
    capA = cv.VideoCapture(VideoA)
    capB = cv.VideoCapture(VideoB)
    
    capA.get(cv.CAP_PROP_FRAME_WIDTH)
    capA.get(cv.CAP_PROP_FRAME_HEIGHT)
    
    # create a cap to contatenated videos
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(VideoConcat, fourcc, FPS, (int(WFrame),  int(HFrame/2)))
    
    
    if (capA.isOpened()== False) or (capB.isOpened()== False): 
        print("Error opening video file")
    
    
    while(capA.isOpened()):
        
    # Capture frame-by-frame
        retA, frameA = capA.read()
        retB, frameB = capB.read()
        
        if retA == True and retB == True:
            
            #concat frames and resize
            frameCat = cv.hconcat([frameA,frameB])
            half = cv.resize(frameCat, (0, 0), fx = 0.5, fy = 0.5)
    
            out.write(half)
        else:
            break
    
    # When everything done, release
    # the video capture object
    capA.release()
    capB.release()
    out.release()
    
    # Closes all the frames
    cv.destroyAllWindows()

def record_vids_sync(VideoA,VideoB,FPS,VideoL,WFrame,HFrame):
    
    t = np.empty((int(VideoL*FPS*1.2),))
    t[:] = np.nan
    t0 = time.time()
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, WFrame)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, HFrame)
    cap.set(cv.CAP_PROP_FPS, int(FPS))
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    
    cap1 = cv.VideoCapture(1, cv.CAP_DSHOW)
    cap1.set(cv.CAP_PROP_FRAME_WIDTH, WFrame)
    cap1.set(cv.CAP_PROP_FRAME_HEIGHT, HFrame)
    cap1.set(cv.CAP_PROP_FPS, int(FPS))
    cap1.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    fourcc1 = cv.VideoWriter_fourcc(*'XVID')
    
    
    out = cv.VideoWriter(VideoA, fourcc, FPS, (WFrame,  HFrame))
    out1 = cv.VideoWriter(VideoB, fourcc1, FPS, (WFrame,  HFrame))
    
    counter = 1;
    
    ## forsome easons the first frames are slower, so we take them out of the loop
    ret, frame = cap.read()
    ret1, frame1 = cap1.read()
    while cap.isOpened():
    
        ## get images from both cameras
        ret, frame = cap.read()
        ret1, frame1 = cap1.read()
    
        if not ret or not ret1:
            print("Can't receive frame (stream end?). Exiting ...")
            break
    
        if counter == 1:  
            t[0] = time.time()
            print(t[0]-t0)
        else:
            t[counter-1] = time.time()-t[0]
            
        if np.mod(counter,2) == 0:
        # contatenate for displaying purposes
            frame_cat = cv.hconcat([frame,frame1])
            half = cv.resize(frame_cat, (0, 0), fx = 0.5, fy = 0.5)
            cv.imshow('frame', half)
            if cv.waitKey(1) != -1:
                break
    
            
        # write the frames in both videos
        out.write(frame)
        out1.write(frame1)
        
        elapsed = time.time() - t[0]
        counter = counter+1
        
        if  elapsed>=VideoL:
            break
    
    # Release everything if job is finished
    cap.release()
    cap1.release()
    
    out.release()
    out1.release()
    cv.destroyAllWindows()
    
    t[0] = 0;
    return t


# In[3]:


## parameters for recording the videos
# Resolution of the cameras
WFrame = 1280
HFrame = 720

# fps
FPS = 30
VideoL = 8 # video length in s

# Video name
Nvideo = '1'
VideoA = '.\Calibration\Calib_video_'+Nvideo+'A.avi'
VideoB = '.\Calibration\Calib_video_'+Nvideo+'B.avi'
VideoConcat = '.\Calibration\Calib_video_'+Nvideo+'concat.avi'
Time_info ='.\Calibration\Calib_video_'+Nvideo+'_t.avi'


# In[4]:


t = record_vids_sync(VideoA,VideoB,FPS,VideoL,WFrame,HFrame)
sidebyside(VideoA,VideoB,VideoConcat)

np.savez(Time_info, t=t)


# In[ ]:




