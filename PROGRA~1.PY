# Import the required libraries
import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import font
from PIL import Image, ImageTk
import cv2

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mediapipe as mp
import time
import imutils
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from scipy import spatial
import os
import math as m
from pathlib import Path
import glob



# Create an instance of tkinter frame or window
win = Tk()
win.title("Perfect pose likes a model")

### Yolo detect Human ###

def Yolo_detect_Human(image):

    # read pre-trained model and config file
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    # read class from coco.name
    classes = []
    with open('coco.names', 'r') as f:
        classes = f.read().splitlines()

    # image = cv2.imread('./Test_img/7.jpg')

    Width = image.shape[1]
    Height = image.shape[0]

    blob = cv2.dnn.blobFromImage(image,1/255,(320,320),(0,0,0), swapRB = True ,crop=False)

    net.setInput(blob)

    Output_layer_name = net.getUnconnectedOutLayersNames()
    layeroutput = net.forward(Output_layer_name)


    class_ids = []
    confidences = []
    boxes = []

    #create bounding box 
    for out in layeroutput:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.7:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([detection[0], detection[1], detection[2], detection[3]])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255,size = (len(boxes),3) )

    blackie = np.zeros(image.shape)

    #check if is people detection

    list_box_human = []
    for i in indices:
    #     print(i)
    #     i = i[0]
        box = boxes[i]
        
        if class_ids[i]==0:

            list_box_human.append(box)
            
    
    return list_box_human


def crop_are_per_Human_0(image_mask, frame, path_save): 
    ## MultiPose
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils
    points = mpPose.PoseLandmark # Landmarks
    data = []
    for p in points:
            x = str(p)[13:]
            data.append(x + "_x")
            data.append(x + "_y")
            data.append(x + "_z")
            data.append(x + "_vis")
    data = pd.DataFrame(columns = data) # Empty dataset


    imgRGB_mask = cv2.cvtColor(image_mask, cv2.COLOR_BGR2RGB)
    results_mask = pose.process(imgRGB_mask)
    lm = results_mask.pose_landmarks

    landmarks_mask = results_mask.pose_landmarks.landmark
    
    i= 0
    temp = []
    while True:
        count = 0
        
    # success, img = cap.read()
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        blackie = np.zeros(frame.shape) # Blank image
        newImage = frame.copy()
        results = pose.process(imgRGB)
        lm_0 = results.pose_landmarks
        
        # print(results.pose_landmarks)
        if results.pose_landmarks:
            # mpDraw.draw_landmarks(blackie, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            mpDraw.draw_landmarks(newImage, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark
             
               
            count_mistake = 0
            for id, lm in enumerate(landmarks) :
                h, w,c = frame.shape
                # print(id, lm)

                temp = temp + [landmarks[id].x, landmarks[id].y, landmarks[id].z, landmarks[id].visibility]
                # print(len(temp))

                list_A = [landmarks[id].x, landmarks[id].y, landmarks[id].z, landmarks[id].visibility]
                # print(list_A)
                # print(type(list_A))
                list_B = [landmarks_mask[id].x, landmarks_mask[id].y, landmarks_mask[id].z, landmarks_mask[id].visibility]

                
                cx, cy = int(landmarks[id].x*w), int(landmarks[id].y*h)
                
                # print(landmarks[id].visibility - landmarks_mask[id].visibility)

                if (abs(landmarks[id].visibility - landmarks_mask[id].visibility)*100)  <= 0.3 :
                    # print(abs(landmarks[id].visibility - landmarks_mask[id].visibility)*100)
                    cv2.circle(newImage, (cx, cy), 3, (255,0,0), cv2.FILLED)

                else :
                    count_mistake += 1
                    cv2.circle(newImage, (cx, cy), 5, (0,0,255), cv2.FILLED)
        
            if count_mistake <= 5:
                
                dir_list = os.listdir(path_save)
                print('yes')
                cv2.imwrite(path_save +'\\img_frame_'+str(len(dir_list)+1)+'.png',frame)
            data.loc[count] = temp
            count +=1
            
        if i == 0 :
            break
    return newImage,temp



# Set the size of the window
win.geometry("1024x850")

# Create two frames in the window
female_button_frame = Frame(win)
male_button_frame = Frame(win)

sit_female_button_frame_page1 = Frame(win)
sit_female_button_frame_page2 = Frame(win)

sit_male_button_frame_page1 = Frame(win)
sit_male_button_frame_page2 = Frame(win)

stand_male_button_frame_page1 = Frame(win)
stand_male_button_frame_page2 = Frame(win)

stand_female_button_frame_page1 = Frame(win)
stand_female_button_frame_page2 = Frame(win)


# Define a function for switching the frames
def change_menu_female():
   
    female_button_frame.pack(fill='both', expand=1)
    male_button_frame.pack_forget()
    
    sit_female_button_frame_page1.pack_forget()
    sit_female_button_frame_page2.pack_forget()
    
    sit_male_button_frame_page1.pack_forget()
    sit_male_button_frame_page2.pack_forget()
    
    stand_male_button_frame_page1.pack_forget()
    stand_male_button_frame_page2.pack_forget()
    
    stand_female_button_frame_page1.pack_forget()
    stand_female_button_frame_page2.pack_forget()
    
    
def change_menu_male():

    male_button_frame.pack(fill='both', expand=1)
    female_button_frame.pack_forget()
    
   
    sit_female_button_frame_page1.pack_forget()
    sit_female_button_frame_page2.pack_forget()
    
    sit_male_button_frame_page1.pack_forget()
    sit_male_button_frame_page2.pack_forget()
    
    stand_male_button_frame_page1.pack_forget()
    stand_male_button_frame_page2.pack_forget()
    
    stand_female_button_frame_page1.pack_forget()
    stand_female_button_frame_page2.pack_forget()
    
### Female ####

def sit_menu_female_page1():
   
    sit_female_button_frame_page1.pack(fill='both', expand=1)
    female_button_frame.pack_forget()
    sit_female_button_frame_page2.pack_forget()
    
def sit_menu_female_page2():
   
    sit_female_button_frame_page2.pack(fill='both', expand=1)
    sit_female_button_frame_page1.pack_forget()
    
    
### Male ####
def sit_menu_male_page1():
   
    sit_male_button_frame_page1.pack(fill='both', expand=1)
    male_button_frame.pack_forget()
    sit_male_button_frame_page2.pack_forget()
    
def sit_menu_male_page2():
   
    sit_male_button_frame_page2.pack(fill='both', expand=1)
    sit_male_button_frame_page1.pack_forget()
    
    
def stand_menu_male_page1():
   
    stand_male_button_frame_page1.pack(fill='both', expand=1)
    male_button_frame.pack_forget()
    stand_male_button_frame_page2.pack_forget()
    
def stand_menu_male_page2():
   
    stand_male_button_frame_page2.pack(fill='both', expand=1)
    stand_male_button_frame_page1.pack_forget()

def stand_menu_female_page1():
   
    stand_female_button_frame_page1.pack(fill='both', expand=1)
    female_button_frame.pack_forget()
    stand_female_button_frame_page2.pack_forget()
    
def stand_menu_female_page2():
   
    stand_female_button_frame_page2.pack(fill='both', expand=1)
    stand_female_button_frame_page1.pack_forget()
    
def show_realtime(path_mask):
    image = cv2.imread(path_mask)
    # print(image.shape)

    list_box = sorted(Yolo_detect_Human(image))
#     print(list_box)

    cam = cv2.VideoCapture(1)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, image.shape[1])
    # print( image.shape[1])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, image.shape[0])     
    # # print( image.shape[0])

    while True:
        (retval,frame)  = cam.read()
        frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
        # frame = cv2.imread('./Template/Women/sit/w_sit_15.jpg')
        
    

        if ( retval ):
            
            
      
            path ='./save_img'
            frame,temp = crop_are_per_Human_0(image,frame,path)
            # print(temp)
            
            

            dim = (600, 800)
            # print(dim)
            frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
            
            for i,j in enumerate(list_box):
                Height = 800
                Width = 600
                
                center_x = int(list_box[i][0] * Width)
                center_y = int(list_box[i][1] * Height)
                w = int(list_box[i][2] * Width)
                h = int(list_box[i][3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                
#                 print(w)
#                 print(h)
#                 print(x)
#                 print(y)
        
                cv2.rectangle(frame, (round(x),round(y)), (round(x+w),round(y+h)), (0, 255, 0), 1)
            
            dir_list = glob.glob("./save_img/*.png")
            dir_list.sort(key=os.path.getmtime)
            # print(dir_list)

            # s_dir_list =dir_list.sort(key=os.path.getmtime)
            # print(s_dir_list)
            

            img_last_path = dir_list[-1]
            print(img_last_path)
            
            img_last = cv2.imread(img_last_path)
            # print(image.shape)
            dim_resize_last_img = ( image.shape[1],400)
            resize_last_img = cv2.resize(img_last, dim_resize_last_img, interpolation = cv2.INTER_AREA)
            # print(resize_last_img.shape)
            
            ## Concat image
            img_concat0 = np.concatenate((image,resize_last_img), axis=0)
            img_concat1 = np.concatenate((img_concat0,frame), axis=1)
            
            cv2.imshow("Pose",img_concat1)
            # cv2.imshow("image",img_3 )
        else:
            print("Error, no image from camera")

        # Wait 1 millisecond for any key press
        if (cv2.waitKey(1)== 27):       # press ESC to quit
            break
    

    
#     label1 = tkinter.Label(newWindow,image=image_mask_test)
#     label1.image = image_mask_test
#     label1.place(x=50, y=50)

    
def get_img_name(path_img):
    show_realtime(path_img)

    
    
#Add a text in Canvas
tk.Label(text = "Perfect pose likes a model ", borderwidth=5, relief="flat", font=("arial italic", 30)).place(x=280,y=20)

# Create fonts for making difference in the frame
font1 = font.Font(family='Georgia', size='16', weight='bold')
font2 = font.Font(family='Aerial', size='12')


#### Female #######
# Menu female
female_button_sit_img = PhotoImage(file='./icon_menu/icon_sit_w.png')
female_button_sit = Button(female_button_frame, image=female_button_sit_img, command=sit_menu_female_page1)
female_button_sit.place(x=220,y=150)
label_female_sit = Label(female_button_frame, text="Sit", foreground="blue", font=font1)
label_female_sit.place(x=270,y=360)

female_button_stand_img = PhotoImage(file='./icon_menu/icon_stand_w.png')
female_button_stand = Button(female_button_frame, image=female_button_stand_img, command=stand_menu_female_page1)
female_button_stand.place(x=430,y=150)
label_female_stand = Label(female_button_frame, text="Stand", foreground="blue", font=font1)
label_female_stand.place(x=460,y=360)


# sit_female_button_frame_page1

female_sit_img0 = PhotoImage(file='./Template/Women/sit/w_sit_0.png')
female_sit0 = Button(sit_female_button_frame_page1, image=female_sit_img0, command=lambda : get_img_name('./Template/Women/sit/w_sit_0_mask.png'))
female_sit0.place(x=220,y=150)

female_sit_img1 = PhotoImage(file='./Template/Women/sit/w_sit_1.png')
female_sit1 = Button(sit_female_button_frame_page1, image=female_sit_img1, command=lambda : get_img_name('./Template/Women/sit/w_sit_1_mask.png'))
female_sit1.place(x=450,y=150)

female_sit_img2 = PhotoImage(file='./Template/Women/sit/w_sit_2.png')
female_sit2 = Button(sit_female_button_frame_page1, image=female_sit_img2, command=lambda : get_img_name('./Template/Women/sit/w_sit_2_mask.png'))
female_sit2.place(x=680,y=150)

female_sit_img3 = PhotoImage(file='./Template/Women/sit/w_sit_3.png')
female_sit3 = Button(sit_female_button_frame_page1, image=female_sit_img3, command=lambda : get_img_name('./Template/Women/sit/w_sit_3_mask.png'))
female_sit3.place(x=220,y=370)

female_sit_img4 = PhotoImage(file='./Template/Women/sit/w_sit_4.png')
female_sit4 = Button(sit_female_button_frame_page1, image=female_sit_img4, command=lambda : get_img_name('./Template/Women/sit/w_sit_4_mask.png'))
female_sit4.place(x=450,y=370)

female_sit_img5 = PhotoImage(file='./Template/Women/sit/w_sit_5.png')
female_sit5 = Button(sit_female_button_frame_page1, image=female_sit_img5, command=lambda : get_img_name('./Template/Women/sit/w_sit_5_mask.png'))
female_sit5.place(x=680,y=370)

female_sit_img6 = PhotoImage(file='./Template/Women/sit/w_sit_6.png')
female_sit6 = Button(sit_female_button_frame_page1, image=female_sit_img6, command=lambda : get_img_name('./Template/Women/sit/w_sit_6_mask.png'))
female_sit6.place(x=250,y=590)

female_sit_img7 = PhotoImage(file='./Template/Women/sit/w_sit_7.png')
female_sit7 = Button(sit_female_button_frame_page1, image=female_sit_img7, command=lambda : get_img_name('./Template/Women/sit/w_sit_7_mask.png'))
female_sit7.place(x=450,y=590)

female_sit_img8 = PhotoImage(file='./Template/Women/sit/w_sit_8.png')
female_sit8 = Button(sit_female_button_frame_page1, image=female_sit_img8, command=lambda : get_img_name('./Template/Women/sit/w_sit_8_mask.png'))
female_sit8.place(x=710,y=590)


nextpage_female = PhotoImage(file='./icon_menu/nextPage.png')
nextpage_button_female = Button(sit_female_button_frame_page1, image=nextpage_female, command=sit_menu_female_page2)
nextpage_button_female.place(x=970,y=800)



# sit_female_button_frame_page2
female_sit_img9 = PhotoImage(file='./Template/Women/sit/w_sit_9.png')
female_sit9 = Button(sit_female_button_frame_page2, image=female_sit_img9, command=lambda : get_img_name('./Template/Women/sit/w_sit_9_mask.png'))
female_sit9.place(x=220,y=150)

female_sit_img10 = PhotoImage(file='./Template/Women/sit/w_sit_10.png')
female_sit10 = Button(sit_female_button_frame_page2, image=female_sit_img10, command=lambda : get_img_name('./Template/Women/sit/w_sit_10_mask.png'))
female_sit10.place(x=450,y=150)

female_sit_img11 = PhotoImage(file='./Template/Women/sit/w_sit_11.png')
female_sit11 = Button(sit_female_button_frame_page2, image=female_sit_img11, command=lambda : get_img_name('./Template/Women/sit/w_sit_11_mask.png'))
female_sit11.place(x=730,y=150)

female_sit_img12 = PhotoImage(file='./Template/Women/sit/w_sit_12.png')
female_sit12 = Button(sit_female_button_frame_page2, image=female_sit_img12, command=lambda : get_img_name('./Template/Women/sit/w_sit_12_mask.png'))
female_sit12.place(x=230,y=370)

female_sit_img13 = PhotoImage(file='./Template/Women/sit/w_sit_13.png')
female_sit13 = Button(sit_female_button_frame_page2, image=female_sit_img13, command=lambda : get_img_name('./Template/Women/sit/w_sit_13_mask.png'))
female_sit13.place(x=480,y=370)

female_sit_img14 = PhotoImage(file='./Template/Women/sit/w_sit_14.png')
female_sit14 = Button(sit_female_button_frame_page2, image=female_sit_img14, command=lambda : get_img_name('./Template/Women/sit/w_sit_14_mask.png'))
female_sit14.place(x=700,y=370)

female_sit_img15 = PhotoImage(file='./Template/Women/sit/w_sit_15.png')
female_sit15 = Button(sit_female_button_frame_page2, image=female_sit_img15, command=lambda : get_img_name('./Template/Women/sit/w_sit_15_mask.png'))
female_sit15.place(x=230,y=590)

female_sit_img16 = PhotoImage(file='./Template/Women/sit/w_sit_16.png')
female_sit16 = Button(sit_female_button_frame_page2, image=female_sit_img16, command=lambda : get_img_name('./Template/Women/sit/w_sit_16_mask.png'))
female_sit16.place(x=470,y=590)


backpage_female = PhotoImage(file='./icon_menu/backpage.png')
backpage_button_female = Button(sit_female_button_frame_page2, image=backpage_female, command=sit_menu_female_page1)
backpage_button_female.place(x=970,y=800)


# stand_female_button_frame_page1
female_stand_img0 = PhotoImage(file='./Template/Women/stand/w_stand_0.png')
female_stand0 = Button(stand_female_button_frame_page1, image=female_stand_img0, command=lambda : get_img_name('./Template/Women/stand/w_stand_0_mask.png'))
female_stand0.place(x=220,y=150)

female_stand_img1 = PhotoImage(file='./Template/Women/stand/w_stand_1.png')
female_stand1 = Button(stand_female_button_frame_page1, image=female_stand_img1, command=lambda : get_img_name('./Template/Women/stand/w_stand_1_mask.png'))
female_stand1.place(x=450,y=150)

female_stand_img2 = PhotoImage(file='./Template/Women/stand/w_stand_2.png')
female_stand2 = Button(stand_female_button_frame_page1, image=female_stand_img2, command=lambda : get_img_name('./Template/Women/stand/w_stand_2_mask.png'))
female_stand2.place(x=680,y=150)

female_stand_img3 = PhotoImage(file='./Template/Women/stand/w_stand_3.png')
female_stand3 = Button(stand_female_button_frame_page1, image=female_stand_img3, command=lambda : get_img_name('./Template/Women/stand/w_stand_3_mask.png'))
female_stand3.place(x=220,y=370)

female_stand_img4 = PhotoImage(file='./Template/Women/stand/w_stand_4.png')
female_stand4 = Button(stand_female_button_frame_page1, image=female_stand_img4, command=lambda : get_img_name('./Template/Women/stand/w_stand_4_mask.png'))
female_stand4.place(x=455,y=370)

female_stand_img5 = PhotoImage(file='./Template/Women/stand/w_stand_5.png')
female_stand5 = Button(stand_female_button_frame_page1, image=female_stand_img5, command=lambda : get_img_name('./Template/Women/stand/w_stand_5_mask.png'))
female_stand5.place(x=680,y=370)

female_stand_img6 = PhotoImage(file='./Template/Women/stand/w_stand_6.png')
female_stand6 = Button(stand_female_button_frame_page1, image=female_stand_img6, command=lambda : get_img_name('./Template/Women/stand/w_stand_6_mask.png'))
female_stand6.place(x=220,y=590)

female_stand_img7 = PhotoImage(file='./Template/Women/stand/w_stand_7.png')
female_stand7 = Button(stand_female_button_frame_page1, image=female_stand_img7, command=lambda : get_img_name('./Template/Women/stand/w_stand_7_mask.png'))
female_stand7.place(x=470,y=590)

female_stand_img8 = PhotoImage(file='./Template/Women/stand/w_stand_8.png')
female_stand8 = Button(stand_female_button_frame_page1, image=female_stand_img8, command=lambda : get_img_name('./Template/Women/stand/w_stand_8_mask.png'))
female_stand8.place(x=680,y=590)


nextpage_female_stand = PhotoImage(file='./icon_menu/nextPage.png')
nextpage_button_female_stand = Button(stand_female_button_frame_page1, image=nextpage_female_stand, command=stand_menu_female_page2)
nextpage_button_female_stand.place(x=970,y=800)


# stand_female_button_frame_page2
female_stand_img9 = PhotoImage(file='./Template/Women/stand/w_stand_9.png')
female_stand9 = Button(stand_female_button_frame_page2, image=female_stand_img9, command=lambda : get_img_name('./Template/Women/stand/w_stand_9_mask.png'))
female_stand9.place(x=220,y=150)

female_stand_img10 = PhotoImage(file='./Template/Women/stand/test.png')
female_stand10 = Button(stand_female_button_frame_page2, image=female_stand_img10, command=lambda : get_img_name('./Template/Women/stand/test_mask.png'))
female_stand10.place(x=450,y=150)



backpage_female_stand = PhotoImage(file='./icon_menu/backpage.png')
backpage_button_female_stand = Button(stand_female_button_frame_page2, image=backpage_female_stand, command=stand_menu_female_page1)
backpage_button_female_stand.place(x=970,y=800)



#### Male #####
# Menu male
male_button_sit_img = PhotoImage(file='./icon_menu/icon_sit_m.png')
male_button_sit = Button(male_button_frame, image=male_button_sit_img, command=sit_menu_male_page1)
male_button_sit.place(x=220,y=150)
label_male_sit = Label(male_button_frame, text="Sit", foreground="blue", font=font1)
label_male_sit.place(x=270,y=360)

male_button_stand_img = PhotoImage(file='./icon_menu/icon_stand_m.png')
male_button_stand = Button(male_button_frame, image=male_button_stand_img, command=stand_menu_male_page1)
male_button_stand.place(x=430,y=150)
label_male_stand = Label(male_button_frame, text="Stand", foreground="blue", font=font1)
label_male_stand.place(x=460,y=360)


# sit_male_button_frame_page1
male_sit_img0 = PhotoImage(file='./Template/Men/sit/m_sit_0.png')
male_sit0 = Button(sit_male_button_frame_page1, image=male_sit_img0, command=lambda : get_img_name('./Template/Men/sit/m_sit_0_mask.png'))
male_sit0.place(x=220,y=150)

male_sit_img1 = PhotoImage(file='./Template/Men/sit/m_sit_1.png')
male_sit1 = Button(sit_male_button_frame_page1, image=male_sit_img1, command=lambda : get_img_name('./Template/Men/sit/m_sit_1_mask.png'))
male_sit1.place(x=450,y=150)

male_sit_img2 = PhotoImage(file='./Template/Men/sit/m_sit_2.png')
male_sit2 = Button(sit_male_button_frame_page1, image=male_sit_img2, command=lambda : get_img_name('./Template/Men/sit/m_sit_2_mask.png'))
male_sit2.place(x=680,y=150)

male_sit_img3 = PhotoImage(file='./Template/Men/sit/m_sit_3.png')
male_sit3 = Button(sit_male_button_frame_page1, image=male_sit_img3, command=lambda : get_img_name('./Template/Men/sit/m_sit_3_mask.png'))
male_sit3.place(x=230,y=370)

male_sit_img4 = PhotoImage(file='./Template/Men/sit/m_sit_4.png')
male_sit4 = Button(sit_male_button_frame_page1, image=male_sit_img4, command=lambda : get_img_name('./Template/Men/sit/m_sit_4_mask.png'))
male_sit4.place(x=470,y=370)

male_sit_img5 = PhotoImage(file='./Template/Men/sit/m_sit_5.png')
male_sit5 = Button(sit_male_button_frame_page1, image=male_sit_img5, command=lambda : get_img_name('./Template/Men/sit/m_sit_5_mask.png'))
male_sit5.place(x=690,y=370)

male_sit_img6 = PhotoImage(file='./Template/Men/sit/m_sit_6.png')
male_sit6 = Button(sit_male_button_frame_page1, image=male_sit_img6, command=lambda : get_img_name('./Template/Men/sit/m_sit_6_mask.png'))
male_sit6.place(x=230,y=590)

male_sit_img7 = PhotoImage(file='./Template/Men/sit/m_sit_7.png')
male_sit7 = Button(sit_male_button_frame_page1, image=male_sit_img7, command=lambda : get_img_name('./Template/Men/sit/m_sit_7_mask.png'))
male_sit7.place(x=480,y=590)

male_sit_img8 = PhotoImage(file='./Template/Men/sit/m_sit_8.png')
male_sit8 = Button(sit_male_button_frame_page1, image=male_sit_img8, command=lambda : get_img_name('./Template/Men/sit/m_sit_8_mask.png'))
male_sit8.place(x=690,y=590)


nextpage_male = PhotoImage(file='./icon_menu/nextPage.png')
nextpage_button_male = Button(sit_male_button_frame_page1, image=nextpage_male, command=sit_menu_male_page2)
nextpage_button_male.place(x=970,y=800)


# sit_male_button_frame_page2
male_sit_img9 = PhotoImage(file='./Template/Men/sit/m_sit_9.png')
male_sit9 = Button(sit_male_button_frame_page2, image=male_sit_img9, command=lambda : get_img_name('./Template/Men/sit/m_sit_9_mask.png'))
male_sit9.place(x=220,y=150)

male_sit_img10 = PhotoImage(file='./Template/Men/sit/m_sit_10.png')
male_sit10 = Button(sit_male_button_frame_page2, image=male_sit_img10, command=lambda : get_img_name('./Template/Men/sit/m_sit_10_mask.png'))
male_sit10.place(x=450,y=150)

male_sit_img12 = PhotoImage(file='./Template/Men/sit/m_sit_12.png')
male_sit12 = Button(sit_male_button_frame_page2, image=male_sit_img12, command=lambda : get_img_name('./Template/Men/sit/m_sit_12_mask.png'))
male_sit12.place(x=700,y=150)

male_sit_img13 = PhotoImage(file='./Template/Men/sit/m_sit_14.png')
male_sit13 = Button(sit_male_button_frame_page2, image=male_sit_img13, command=lambda : get_img_name('./Template/Men/sit/m_sit_14_mask.png'))
male_sit13.place(x=220,y=370)

male_sit_img14 = PhotoImage(file='./Template/Men/sit/m_sit_15.png')
male_sit14 = Button(sit_male_button_frame_page2, image=male_sit_img14, command=lambda : get_img_name('./Template/Men/sit/m_sit_15_mask.png'))
male_sit14.place(x=450,y=370)

male_sit_img15 = PhotoImage(file='./Template/Men/sit/m_sit_13.png')
male_sit15 = Button(sit_male_button_frame_page2, image=male_sit_img15, command=lambda : get_img_name('./Template/Men/sit/m_sit_13_mask.png'))
male_sit15.place(x=220,y=590)

backpage_male = PhotoImage(file='./icon_menu/backpage.png')
backpage_button_male = Button(sit_male_button_frame_page2, image=backpage_male, command=sit_menu_male_page1)
backpage_button_male.place(x=970,y=800)

# stand_male_button_frame_page1
male_stand_img0 = PhotoImage(file='./Template/Men/stand/m_stand_0.png')
male_stand0 = Button(stand_male_button_frame_page1, image=male_stand_img0, command=lambda : get_img_name('./Template/Men/stand/m_stand_0_mask.png'))
male_stand0.place(x=220,y=150)

male_stand_img1 = PhotoImage(file='./Template/Men/stand/m_stand_1.png')
male_stand1 = Button(stand_male_button_frame_page1, image=male_stand_img1, command=lambda : get_img_name('./Template/Men/stand/m_stand_1_mask.png'))
male_stand1.place(x=450,y=150)

male_stand_img2 = PhotoImage(file='./Template/Men/stand/m_stand_2.png')
male_stand2 = Button(stand_male_button_frame_page1, image=male_stand_img2, command=lambda : get_img_name('./Template/Men/stand/m_stand_2_mask.png'))
male_stand2.place(x=680,y=150)

male_stand_img3 = PhotoImage(file='./Template/Men/stand/m_stand_3.png')
male_stand3 = Button(stand_male_button_frame_page1, image=male_stand_img3, command=lambda : get_img_name('./Template/Men/stand/m_stand_3_mask.png'))
male_stand3.place(x=230,y=370)

male_stand_img4 = PhotoImage(file='./Template/Men/stand/m_stand_4.png')
male_stand4 = Button(stand_male_button_frame_page1, image=male_stand_img4, command=lambda : get_img_name('./Template/Men/stand/m_stand_4_mask.png'))
male_stand4.place(x=450,y=370)

male_stand_img5 = PhotoImage(file='./Template/Men/stand/m_stand_5.png')
male_stand5 = Button(stand_male_button_frame_page1, image=male_stand_img5, command=lambda : get_img_name('./Template/Men/stand/m_stand_5_mask.png'))
male_stand5.place(x=690,y=370)

male_stand_img6 = PhotoImage(file='./Template/Men/stand/m_stand_6.png')
male_stand6 = Button(stand_male_button_frame_page1, image=male_stand_img6, command=lambda : get_img_name('./Template/Men/stand/m_stand_6_mask.png'))
male_stand6.place(x=220,y=590)

male_stand_img7 = PhotoImage(file='./Template/Men/stand/m_stand_7.png')
male_stand7 = Button(stand_male_button_frame_page1, image=male_stand_img7, command=lambda : get_img_name('./Template/Men/stand/m_stand_7_mask.png'))
male_stand7.place(x=440,y=590)

male_stand_img8 = PhotoImage(file='./Template/Men/stand/m_stand_8.png')
male_stand8 = Button(stand_male_button_frame_page1, image=male_stand_img8, command=lambda : get_img_name('./Template/Men/stand/m_stand_8_mask.png'))
male_stand8.place(x=690,y=590)


nextpage_male_stand = PhotoImage(file='./icon_menu/nextPage.png')
nextpage_button_male_stand = Button(stand_male_button_frame_page1, image=nextpage_male_stand, command=stand_menu_male_page2)
nextpage_button_male_stand.place(x=970,y=800)


# stand_female_button_frame_page2
male_stand_img9 = PhotoImage(file='./Template/Men/stand/m_stand_9.png')
male_stand9 = Button(stand_male_button_frame_page2, image=male_stand_img9, command=lambda : get_img_name('./Template/Men/stand/m_stand_9_mask.png'))
male_stand9.place(x=220,y=150)

male_stand_img10 = PhotoImage(file='./Template/Men/stand/m_stand_10.png')
male_stand10 = Button(stand_male_button_frame_page2, image=male_stand_img10, command=lambda : get_img_name('./Template/Men/stand/m_stand_10_mask.png'))
male_stand10.place(x=450,y=150)

male_stand_img11 = PhotoImage(file='./Template/Men/stand/m_stand_11.png')
male_stand11 = Button(stand_male_button_frame_page2, image=male_stand_img11, command=lambda : get_img_name('./Template/Men/stand/m_stand_11_mask.png'))
male_stand11.place(x=700,y=150)

male_stand_img12 = PhotoImage(file='./Template/Men/stand/m_stand_12.png')
male_stand12 = Button(stand_male_button_frame_page2, image=male_stand_img12, command=lambda : get_img_name('./Template/Men/stand/m_stand_12_mask.png'))
male_stand12.place(x=220,y=370)

male_stand_img13 = PhotoImage(file='./Template/Men/stand/m_stand_13.png')
male_stand13 = Button(stand_male_button_frame_page2, image=male_stand_img13, command=lambda : get_img_name('./Template/Men/stand/m_stand_13_mask.png'))
male_stand13.place(x=450,y=370)

male_stand_img14 = PhotoImage(file='./Template/Men/stand/m_stand_14.png')
male_stand14 = Button(stand_male_button_frame_page2, image=male_stand_img14, command=lambda : get_img_name('./Template/Men/stand/m_stand_14_mask.png'))
male_stand14.place(x=690,y=370)



backpage_male_stand = PhotoImage(file='./icon_menu/backpage.png')
backpage_button_male_stand = Button(stand_male_button_frame_page2, image=backpage_male_stand, command=stand_menu_male_page1)
backpage_button_male_stand.place(x=970,y=800)


#### Quit button ###
Quit_button_img = PhotoImage(file='./icon_menu/quit.png')
Quit_button = Button(win, image=Quit_button_img, command=win.destroy)
Quit_button.place(x=25,y=500)


# Add a button to switch between two frames
female_button_img = PhotoImage(file='./icon_menu/female_0.png')
female_button = Button(win, image=female_button_img, command=change_menu_female)
female_button.place(x=20,y=150)

male_button_img = PhotoImage(file='./icon_menu/male.png')
male_button = Button(win, image=male_button_img, command=change_menu_male)
male_button.place(x=100,y=150)

win.mainloop()