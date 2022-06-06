import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mediapipe as mp
import time
import imutils

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


def crop_are_per_Human(list_box_human):
    start_y = int(list_box_human[1] )      # value between 0 to ( img.shape[0]-1 )
    end_y = int( list_box_human[1] +list_box_human[3] )         # value between 0 to ( img.shape[0]-1 )
    

    start_x = int(list_box_human[0]  )      # value between 0 to ( img.shape[1]-1 )
    end_x = int(list_box_human[0] +list_box_human[2])       # value between 0 to ( img.shape[1]-1 )

    crop = image[ start_y:end_y , start_x:end_x ]
    # cv2.imshow("CROP",crop)
    # cv2.waitKey(0)

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
        imgRGB = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        blackie = np.zeros(crop.shape) # Blank image

        newImage = crop.copy()


        results = pose.process(imgRGB)

        # print(results.pose_landmarks)
        if results.pose_landmarks:
            mpDraw.draw_landmarks(blackie, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            mpDraw.draw_landmarks(newImage, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark

            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w,c = crop.shape
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
        
        image[ start_y:end_y , start_x:end_x ]= newImage

        # x = list_box_human[0]
        # y = list_box_human[1]
        # w = list_box_human[2]
        # h = list_box_human[3]
        
        # cv2.rectangle(image, (round(x),round(y)), (round(x+w),round(y+h)), (0, 255, 0), 2)
            
        # cv2.imshow("CROP OVERLAY",image)
        # cv2.waitKey(0)      # flush image buffer

        if i == 0 :
            break

    return image,temp












## Check Yolo detect Human Function
image = cv2.imread('./Image_Test/7.jpg')
print(image.shape)
list_box , shape_img =Yolo_detect_Human(image)
# print(sorted(list_box))
print(shape_img)


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
print(data.columns)

Total_landmask = []
for i,j in enumerate(sorted(list_box)):
    img,temp = crop_are_per_Human(list_box[i])
    Total_landmask.append(temp)

print(len(list_box))
print(list_box)

data = pd.DataFrame(Total_landmask,columns=data.columns)
print(img.shape)

# cv2.imshow("img",img)
# cv2.waitKey(0)








cam = cv2.VideoCapture(0)  
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)     

while True:
    retval,frame  = cam.read()
  
  
    if ( retval ):
        for i,j in enumerate(list_box):
            x = list_box[i][0]
            y = list_box[i][1]
            w = list_box[i][2]
            h = list_box[i][3]
        
            cv2.rectangle(frame, (round(x),round(y)), (round(x+w),round(y+h)), (0, 255, 0), 2)
            
        
        list_box_frame , shape_img_frame =Yolo_detect_Human(frame)
        print(list_box_frame)

        Total_landmask_frame = []
        for i,j in enumerate(sorted(list_box_frame)):
            img_frame,temp_frame = crop_are_per_Human(list_box_frame[i])
            Total_landmask_frame.append(temp_frame)


        cv2.imshow("img_frame",img_frame)
        # cv2.imshow("img",img)
        cv2.imshow("Camera",frame )
    else:
        print("Error, no image from camera")

    # Wait 1 millisecond for any key press
    if (cv2.waitKey(1)== 27):       # press ESC to quit
        break

