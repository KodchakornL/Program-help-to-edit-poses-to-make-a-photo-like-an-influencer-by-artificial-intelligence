import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mediapipe as mp
import time
import imutils
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from scipy import spatial
import os
import math as m


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
                boxes.append([x, y, w, h])


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
            
            # x = box[0]
            # y = box[1]
            # w = box[2]
            # h = box[3]
            # label = str(classes[class_id]) 
            
            # cv2.rectangle(blackie, (round(x),round(y)), (round(x+w),round(y+h)), (0, 255, 0), 2)
            # cv2.putText(blackie, label, (round(box[0])-10,round(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # print(list_box_human)
    # cv2.imshow("blackie", blackie)
    # cv2.imshow("image", image)

    # cv2.waitKey(0)
    
    return list_box_human,image.shape


def crop_are_per_Human(frame):

    ## MultiPose

    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils
    points = mpPose.PoseLandmark # Landmarks

    pTime = 0

    data = []
    for p in points:
            x = str(p)[13:]
            data.append(x + "_x")
            data.append(x + "_y")
            data.append(x + "_z")
            data.append(x + "_vis")
    data = pd.DataFrame(columns = data) # Empty dataset


    i= 0
    temp = []
    while True:
        count = 0
        
    # success, img = cap.read()
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        blackie = np.zeros(frame.shape) # Blank image

        newImage = frame.copy()


        results = pose.process(imgRGB)

        # print(results.pose_landmarks)
        if results.pose_landmarks:
            mpDraw.draw_landmarks(blackie, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            mpDraw.draw_landmarks(newImage, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark

            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w,c = frame.shape
                # print(id, lm)

                temp = temp + [lm.x, lm.y, lm.z, lm.visibility]
                # print(len(temp))

                cx, cy = int(lm.x*w), int(lm.y*h)

                cv2.circle(blackie, (cx, cy), 5, (255,0,0), cv2.FILLED)
                cv2.circle(newImage, (cx, cy), 5, (255,0,0), cv2.FILLED)

            data.loc[count] = temp
            count +=1
            data.to_csv("dataset3.csv") # save the data as a csv file


        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        # cv2.imshow("newImage", newImage)

        # cv2.imshow("blackie",blackie)

        # cv2.putText(crop, str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
        # cv2.imshow("Image", crop)
        # cv2.waitKey(0)



        # Copy 'crop' and place it at the top-left corner of the original image
        # print(blackie.shape)
        # print(crop.shape)
        
    

        # x = list_box_human[0]
        # y = list_box_human[1]
        # w = list_box_human[2]
        # h = list_box_human[3]
        
        # cv2.rectangle(image, (round(x),round(y)), (round(x+w),round(y+h)), (0, 255, 0), 2)
            
        # cv2.imshow("CROP OVERLAY",image)
        # cv2.waitKey(0)      # flush image buffer

        if i == 0 :
            break

    return newImage,temp


def cosine_similarity(list_A,List_B):
   
    result= 1 - spatial.distance.cosine(list_A, List_B)
    return result

# Calculate angle.

def findAngle(x1, y1, x2, y2):
    theta = m.acos( (y2 -y1)*(-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2 ) * y1) )

    degree = int(180/m.pi)*theta

    return degree



def crop_are_per_Human_0(image_mask, frame, path_save):
    
    
    ## MultiPose

    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils
    points = mpPose.PoseLandmark # Landmarks

    pTime = 0

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
    # print(lm)

    # # LEFT_EYE_INNER
    # l_EYE_INNER_x = lm.landmark[points.LEFT_EYE_INNER].x 
    # l_EYE_INNER_y = lm.landmark[points.LEFT_EYE_INNER].y 

    # # RIGHT_EYE_INNER
    # R_EYE_INNER_x = lm.landmark[points.RIGHT_EYE_INNER].x 
    # R_EYE_INNER_y = lm.landmark[points.RIGHT_EYE_INNER].y 

   

    # # LEFT_EYE
    # l_LEFT_EYE_x = int(lm.landmark[points.LEFT_EYE].x )
    # l_LEFT_EYE_y = int(lm.landmark[points.LEFT_EYE].y )

    # # RIGHT_EYE
    # R_RIGHT_EYE_x = int(lm.landmark[points.RIGHT_EYE].x )
    # R_RIGHT_EYE_y = int(lm.landmark[points.RIGHT_EYE].y )

    # # LEFT_EYE_OUTER
    # l_LEFT_EYE_OUTER_x = int(lm.landmark[points.LEFT_EYE_OUTER].x )
    # l_LEFT_EYE_OUTER_y = int(lm.landmark[points.LEFT_EYE_OUTER].y )

    # # RIGHT_EYE
    # R_RIGHT_EYE_OUTER_x = int(lm.landmark[points.RIGHT_EYE_OUTER].x )
    # R_RIGHT_EYE_OUTER_y = int(lm.landmark[points.RIGHT_EYE_OUTER].y )

    # # LEFT_EYE_OUTER
    # l_LEFT_EAR_x = int(lm.landmark[points.LEFT_EAR].x )
    # l_LEFT_EAR_y = int(lm.landmark[points.LEFT_EAR].y )

    # # RIGHT_EYE
    # R_RIGHT_EAR_x = int(lm.landmark[points.RIGHT_EAR].x )
    # R_RIGHT_EAR_y = int(lm.landmark[points.RIGHT_EAR].y )

    # # MOUTH_LEFT
    # l_MOUTH_LEFT_x = int(lm.landmark[points.MOUTH_LEFT].x )
    # l_MOUTH_LEFT_y = int(lm.landmark[points.MOUTH_LEFT].y )

    # # MOUTH_RIGHT
    # R_MOUTH_RIGHT_x = int(lm.landmark[points.MOUTH_RIGHT].x )
    # R_MOUTH_RIGHT_y = int(lm.landmark[points.MOUTH_RIGHT].y )

    # # RIGHT_SHOULDER
    # l_LEFT_SHOULDER_x = lm.landmark[points.LEFT_SHOULDER].x 
    # l_LEFT_SHOULDER_y = lm.landmark[points.LEFT_SHOULDER].y 

    # # RIGHT_SHOULDER
    # R_RIGHT_SHOULDER_x = lm.landmark[points.RIGHT_SHOULDER].x 
    # R_RIGHT_SHOULDER_y = lm.landmark[points.RIGHT_SHOULDER].y 

    # angle_shoulder = findAngle(l_LEFT_SHOULDER_x, l_LEFT_SHOULDER_y, R_RIGHT_SHOULDER_x, R_RIGHT_SHOULDER_y)

    # # LEFT_ELBOW
    # l_LEFT_ELBOW_x = lm.landmark[points.LEFT_ELBOW].x 
    # l_LEFT_ELBOW_y = lm.landmark[points.LEFT_ELBOW].y 

    # # RIGHT_ELBOW
    # R_RIGHT_ELBOW_x = lm.landmark[points.RIGHT_ELBOW].x 
    # R_RIGHT_ELBOW_y = lm.landmark[points.RIGHT_ELBOW].y 

    # angle_elbow = findAngle(l_LEFT_ELBOW_x, l_LEFT_ELBOW_y, R_RIGHT_ELBOW_x, R_RIGHT_ELBOW_y)

    # # LEFT_WRIST
    # l_LEFT_WRIST_x = lm.landmark[points.LEFT_ELBOW].x 
    # l_LEFT_WRIST_y = lm.landmark[points.LEFT_ELBOW].y 

    # # RIGHT_WRIST
    # R_RIGHT_WRIST_x = lm.landmark[points.RIGHT_WRIST].x 
    # R_RIGHT_WRIST_y = lm.landmark[points.RIGHT_WRIST].y 

    # angle_wrist = findAngle(l_LEFT_WRIST_x, l_LEFT_WRIST_y, R_RIGHT_WRIST_x, R_RIGHT_WRIST_y)

    # # LEFT_PINKY
    # l_LEFT_PINKY_x = lm.landmark[points.LEFT_PINKY].x 
    # l_LEFT_PINKY_y = lm.landmark[points.LEFT_PINKY].y 

    # # RIGHT_PINKY
    # R_RIGHT_PINKY_x = lm.landmark[points.RIGHT_PINKY].x 
    # R_RIGHT_PINKY_y = lm.landmark[points.RIGHT_PINKY].y 

    # angle_pinky = findAngle(l_LEFT_PINKY_x, l_LEFT_PINKY_y, R_RIGHT_PINKY_x, R_RIGHT_PINKY_y)

    # # LEFT_INDEX
    # l_LEFT_INDEX_x = lm.landmark[points.LEFT_INDEX].x 
    # l_LEFT_INDEX_y = lm.landmark[points.LEFT_INDEX].y 

    # # RIGHT_INDEX
    # R_RIGHT_INDEX_x = lm.landmark[points.RIGHT_INDEX].x 
    # R_RIGHT_INDEX_y = lm.landmark[points.RIGHT_INDEX].y 

    # angle_index = findAngle(l_LEFT_INDEX_x, l_LEFT_INDEX_y, R_RIGHT_INDEX_x, R_RIGHT_INDEX_y)

    # # LEFT_THUMB
    # l_LEFT_THUMB_x = lm.landmark[points.LEFT_THUMB].x 
    # l_LEFT_THUMB_y = lm.landmark[points.LEFT_THUMB].y 

    # # RIGHT_THUMB
    # R_RIGHT_THUMB_x = lm.landmark[points.RIGHT_THUMB].x 
    # R_RIGHT_THUMB_y = lm.landmark[points.RIGHT_THUMB].y 

    # angle_thumb = findAngle(l_LEFT_THUMB_x, l_LEFT_THUMB_y, R_RIGHT_THUMB_x, R_RIGHT_THUMB_y)

    # # LEFT_HIP
    # l_LEFT_HIP_x = lm.landmark[points.LEFT_HIP].x 
    # l_LEFT_HIP_y = lm.landmark[points.LEFT_HIP].y 

    # # RIGHT_HIP
    # R_RIGHT_HIP_x = lm.landmark[points.RIGHT_HIP].x 
    # R_RIGHT_HIP_y = lm.landmark[points.RIGHT_HIP].y 

    # angle_hip = findAngle(l_LEFT_HIP_x, l_LEFT_HIP_y, R_RIGHT_HIP_x, R_RIGHT_HIP_y)

    # # LEFT_KNEE
    # l_LEFT_KNEE_x = lm.landmark[points.LEFT_KNEE].x 
    # l_LEFT_KNEE_y = lm.landmark[points.LEFT_KNEE].y 

    # # RIGHT_KNEE
    # R_RIGHT_KNEE_x = lm.landmark[points.RIGHT_KNEE].x 
    # R_RIGHT_KNEE_y = lm.landmark[points.RIGHT_KNEE].y 

    # angle_knee = findAngle(l_LEFT_KNEE_x, l_LEFT_KNEE_y, R_RIGHT_KNEE_x, R_RIGHT_KNEE_y)

    # # LEFT_ANKLE
    # l_LEFT_ANKLE_x = lm.landmark[points.LEFT_ANKLE].x 
    # l_LEFT_ANKLE_y = lm.landmark[points.LEFT_ANKLE].y 

    # # RIGHT_ANKLE
    # R_RIGHT_ANKLE_x = lm.landmark[points.RIGHT_ANKLE].x 
    # R_RIGHT_ANKLE_y = lm.landmark[points.RIGHT_ANKLE].y 

    # angle_ankle = findAngle(l_LEFT_ANKLE_x, l_LEFT_ANKLE_y, R_RIGHT_ANKLE_x, R_RIGHT_ANKLE_y)

    # # LEFT_HEEL
    # l_LEFT_HEEL_x = lm.landmark[points.LEFT_HEEL].x 
    # l_LEFT_HEEL_y = lm.landmark[points.LEFT_HEEL].y 

    # # RIGHT_HEEL
    # R_RIGHT_HEEL_x = lm.landmark[points.RIGHT_HEEL].x 
    # R_RIGHT_HEEL_y = lm.landmark[points.RIGHT_HEEL].y 

    # angle_heel = findAngle(l_LEFT_HEEL_x, l_LEFT_HEEL_y, R_RIGHT_HEEL_x, R_RIGHT_HEEL_y)

    # # LEFT_FOOT_INDEX
    # l_LEFT_FOOT_INDEX_x = lm.landmark[points.LEFT_FOOT_INDEX].x 
    # l_LEFT_FOOT_INDEX_y = lm.landmark[points.LEFT_FOOT_INDEX].y 

    # # RIGHT_FOOT_INDEX
    # R_RIGHT_FOOT_INDEX_x = lm.landmark[points.RIGHT_FOOT_INDEX].x 
    # R_RIGHT_FOOT_INDEXL_y = lm.landmark[points.RIGHT_FOOT_INDEX].y

    # angle_foot = findAngle(l_LEFT_FOOT_INDEX_x, l_LEFT_FOOT_INDEX_y, R_RIGHT_FOOT_INDEX_x, R_RIGHT_FOOT_INDEXL_y)

    # list_angle = [angle_shoulder , angle_elbow,angle_wrist,angle_pinky,angle_index,angle_thumb,angle_hip,angle_knee,angle_ankle,angle_heel,angle_foot]
    # print(list_angle)
    # print(len(list_angle))
    # print('\n')



    

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
             
            # # RIGHT_SHOULDER
            # l_LEFT_SHOULDER_x_0 = int(lm_0.landmark[points.LEFT_SHOULDER].x )
            # l_LEFT_SHOULDER_y_0 = int(lm_0.landmark[points.LEFT_SHOULDER].y )

            # # RIGHT_SHOULDER
            # R_RIGHT_SHOULDER_x_0 = int(lm_0.landmark[points.RIGHT_SHOULDER].x )
            # R_RIGHT_SHOULDER_y_0 = int(lm_0.landmark[points.RIGHT_SHOULDER].y )

            # # LEFT_ELBOW
            # l_LEFT_ELBOW_x_0 = int(lm_0.landmark[points.LEFT_ELBOW].x )
            # l_LEFT_ELBOW_y_0 = int(lm_0.landmark[points.LEFT_ELBOW].y )

            # # RIGHT_ELBOW
            # R_RIGHT_ELBOW_x_0 = int(lm_0.landmark[points.RIGHT_ELBOW].x )
            # R_RIGHT_ELBOW_y_0 = int(lm_0.landmark[points.RIGHT_ELBOW].y )

            # # LEFT_WRIST
            # l_LEFT_WRIST_x_0 = int(lm_0.landmark[points.LEFT_ELBOW].x )
            # l_LEFT_WRIST_y_0 = int(lm_0.landmark[points.LEFT_ELBOW].y )

            # # RIGHT_WRIST
            # R_RIGHT_WRIST_x_0 = int(lm_0.landmark[points.RIGHT_WRIST].x )
            # R_RIGHT_WRIST_y_0 = int(lm_0.landmark[points.RIGHT_WRIST].y )

            # # LEFT_PINKY
            # l_LEFT_PINKY_x_0 = int(lm_0.landmark[points.LEFT_PINKY].x )
            # l_LEFT_PINKY_y_0 = int(lm_0.landmark[points.LEFT_PINKY].y )

            # # RIGHT_PINKY
            # R_RIGHT_PINKY_x_0 = int(lm_0.landmark[points.RIGHT_PINKY].x )
            # R_RIGHT_PINKY_y_0 = int(lm_0.landmark[points.RIGHT_PINKY].y )

            # # LEFT_INDEX
            # l_LEFT_INDEX_x_0 = int(lm_0.landmark[points.LEFT_INDEX].x )
            # l_LEFT_INDEX_y_0 = int(lm_0.landmark[points.LEFT_INDEX].y )

            # # RIGHT_INDEX
            # R_RIGHT_INDEX_x = int(lm_0.landmark[points.RIGHT_INDEX].x )
            # R_RIGHT_INDEX_y = int(lm_0.landmark[points.RIGHT_INDEX].y )

            # # LEFT_THUMB
            # l_LEFT_THUMB_x_0 = int(lm_0.landmark[points.LEFT_THUMB].x )
            # l_LEFT_THUMB_y_0 = int(lm_0.landmark[points.LEFT_THUMB].y )

            # # RIGHT_THUMB
            # R_RIGHT_THUMB_x_0 = int(lm_0.landmark[points.RIGHT_THUMB].x )
            # R_RIGHT_THUMB_y_0 = int(lm_0.landmark[points.RIGHT_THUMB].y )

            # # LEFT_HIP
            # l_LEFT_HIP_x_0 = int(lm_0.landmark[points.LEFT_HIP].x )
            # l_LEFT_HIP_y_0 = int(lm_0.landmark[points.LEFT_HIP].y )

            # # RIGHT_HIP
            # R_RIGHT_HIP_x_0 = int(lm_0.landmark[points.RIGHT_HIP].x )
            # R_RIGHT_HIP_y_0 = int(lm_0.landmark[points.RIGHT_HIP].y )

            # # LEFT_KNEE
            # l_LEFT_KNEE_x_0 = int(lm_0.landmark[points.LEFT_KNEE].x )
            # l_LEFT_KNEE_y_0 = int(lm_0.landmark[points.LEFT_KNEE].y )

            # # RIGHT_KNEE
            # R_RIGHT_KNEE_x_0 = int(lm_0.landmark[points.RIGHT_KNEE].x )
            # R_RIGHT_KNEE_y_0 = int(lm_0.landmark[points.RIGHT_KNEE].y )

            # # LEFT_ANKLE
            # l_LEFT_ANKLE_x_0 = int(lm_0.landmark[points.LEFT_ANKLE].x )
            # l_LEFT_ANKLE_y_0 = int(lm_0.landmark[points.LEFT_ANKLE].y )

            # # RIGHT_ANKLE
            # R_RIGHT_ANKLE_x_0 = int(lm_0.landmark[points.RIGHT_ANKLE].x )
            # R_RIGHT_ANKLE_y_0 = int(lm.landmark[points.RIGHT_ANKLE].y )

            # # LEFT_HEEL
            # l_LEFT_HEEL_x_0 = int(lm_0.landmark[points.LEFT_HEEL].x )
            # l_LEFT_HEEL_y_0 = int(lm_0.landmark[points.LEFT_HEEL].y )

            # # RIGHT_HEEL
            # R_RIGHT_HEEL_x_0 = int(lm_0.landmark[points.RIGHT_HEEL].x )
            # R_RIGHT_HEEL_y_0 = int(lm_0.landmark[points.RIGHT_HEEL].y )

            # # LEFT_FOOT_INDEX
            # l_LEFT_FOOT_INDEX_x_0 = int(lm_0.landmark[points.LEFT_FOOT_INDEX].x )
            # l_LEFT_FOOT_INDEX_y_0 = int(lm_0.landmark[points.LEFT_FOOT_INDEX].y )

            # # RIGHT_FOOT_INDEX
            # R_RIGHT_FOOT_INDEX_x_0 = int(lm_0.landmark[points.RIGHT_FOOT_INDEX].x )
            # R_RIGHT_FOOT_INDEXL_y_0 = int(lm_0.landmark[points.RIGHT_FOOT_INDEX].y )
                    
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
                # print(id)
                # if (id in range(11,33)) and (id % 2 == 0):
                    
                #     print('\n')
                #     angle = findAngle(landmarks[id-1].x, landmarks[id-1].y, landmarks[id].x, landmarks[id].y)
                #     print(id)
                #     print(f' 2; {id-12}')
                    
                #     if angle >= list_angle[id-11]*0.95 :
                #         cv2.circle(newImage, (cx, cy), 5, (255,0,0), cv2.FILLED)
                #     else:
                #         cv2.circle(newImage, (cx, cy), 10, (0,0,255), cv2.FILLED)
                # elif (id in range(11,33)) and (id % 2 == 1) :
                #     print(id)
                #     print(f' 1; {id-11}')
                #     angle = findAngle(landmarks[id].x, landmarks[id].y, landmarks[id+1].x, landmarks[id+1].y)
                #     if angle >= list_angle[id-11]*0.95 :
                #         cv2.circle(newImage, (cx, cy), 5, (255,0,0), cv2.FILLED)
                #     else:
                #         cv2.circle(newImage, (cx, cy), 10, (0,0,255), cv2.FILLED)

                if (landmarks[id].visibility  >= landmarks_mask[id].visibility*0.9999)   :
                    # print(landmarks[id].visibility)
                    # print(landmarks_mask[id].visibility)
                    cv2.circle(newImage, (cx, cy), 5, (255,0,0), cv2.FILLED)

                else :
                    count_mistake += 1
                    # print(count_mistake)
                    
                    cv2.circle(newImage, (cx, cy), 10, (0,0,255), cv2.FILLED)
        
            if count_mistake == 0:
                
                dir_list = os.listdir(path_save)
                # print(len(dir_list))
                print('yes')
                cv2.imwrite(path_save +'\\img_frame'+str(len(dir_list)+1)+'.jpg',frame)
            data.loc[count] = temp
            count +=1
            data.to_csv("dataset3.csv") # save the data as a csv file


        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        # cv2.imshow("newImage", newImage)

        # cv2.imshow("blackie",blackie)

        # cv2.putText(crop, str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
        # cv2.imshow("Image", crop)
        # cv2.waitKey(0)



        # Copy 'crop' and place it at the top-left corner of the original image
        # print(blackie.shape)
        # print(crop.shape)
        
    

        # x = list_box_human[0]
        # y = list_box_human[1]
        # w = list_box_human[2]
        # h = list_box_human[3]
        
        # cv2.rectangle(image, (round(x),round(y)), (round(x+w),round(y+h)), (0, 255, 0), 2)
            
        # cv2.imshow("CROP OVERLAY",image)
        # cv2.waitKey(0)      # flush image buffer

        if i == 0 :
            break

    return newImage,temp










## Check Yolo detect Human Function
image = cv2.imread('./Template/Women/sit/w_sit_14.jpeg')


scale_percent = 180 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
print(dim)
image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
# list_box , shape_img =Yolo_detect_Human(image)

image,temp = crop_are_per_Human(image)






cam = cv2.VideoCapture(0)  
cam.set(cv2.CAP_PROP_FRAME_WIDTH, image.shape[1])
# print( image.shape[1])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, image.shape[0])     
# print( image.shape[0])

while True:
    retval,frame  = cam.read()

    frame = cv2.imread('./Template/Women/sit/w_sit_15.jpeg')
  
  
    if ( retval ):
        # for i,j in enumerate(list_box):
        #     x = list_box[i][0]
        #     y = list_box[i][1]
        #     w = list_box[i][2]
        #     h = list_box[i][3]
        
            # cv2.rectangle(frame, (round(x),round(y)), (round(x+w),round(y+h)), (0, 255, 0), 2)
            
        path ='./save_img'
        frame,temp = crop_are_per_Human_0(image,frame,path)
        # print(temp)

    
        cv2.imshow("image_frame",frame)
        # cv2.imshow("img",img)
        cv2.imshow("image",image )
    else:
        print("Error, no image from camera")

    # Wait 1 millisecond for any key press
    if (cv2.waitKey(1)== 27):       # press ESC to quit
        break

