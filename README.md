# Program help to edit poses to make a photo like an influencer by artificial intelligence
Project  in BADS7203 Image and Video Analytics  
[![](https://img.shields.io/badge/-BlazePose-blue)](#) [![](https://img.shields.io/badge/-Tiny--YOLO--V3--Algorithm-green)](#)  

## Introduction
Taking a photo is a recording of the impression that happened. Getting good or bad photos affects the state of mind that poses for a photo is not everyone's favorite thing. The creator therefore created a program to help photographers take photos according to the poses selected by the model or model chosen by the pose using the Blazepose model to detect the correct poses from the creation. 33 key points to get the right pose The key points from Blazepose help to tell which point is wrong with the color of the point. If the pose is wrong, Key points will be red. When the pose is correct, Key points will be blue, allowing the pose to be properly positioned. And it can also work well with real-time work. It is used in conjunction with the YOLO V3 algorithm to detect poses by creating a frame to indicate where the poses are located. When the poser is able to pose correctly, all will automatically take a photo and save it.  

## Research Methodology  
Creating a program to help edit poses to get an influencer-like image with artificial intelligence.The technique used is Human Pose Estimation and Posing Detection with deep learning by BlazePose Algorithm to detect the correct pose and take photos automatically when The pose is correct according to the key points created on the body.

The dataset used image data published from the Printerest app by selecting a model image consisting of sitting and standing. This program is developed with specs Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz 1.99 GHz. Library used and version includes  
1.) Tkinter version 0.3.1  
2.) cv2 (Opencv) version 4.5.5  
3.) numpy version 1.21.4  
4.) pandas version 1.0.5  
5.) mediapipe version 0.8.10  
6.) time version 1.0.0  
7.) os  
the program with 2 main parts:
1.) Create a Graphical user interface (GUI) so that program users can choose the poses they want.  
  
**Step 1. Choose gender between male and female.**  
  
![step1](./slide_ppt/Picture1.1.png =250x250)  
  
**Step 2. Choose a posture between sitting and standing. Examples :**  
  
![step2](./slide_ppt/Picture1.2.png)  
![step2](./slide_ppt/Picture1.3.png)  
  
**Step 3. Select the desired pose. as in the examples :**  
  
![step3](./slide_ppt/Picture1.4.png)  
![step3](./slide_ppt/Picture1.5.png)  
![step3](./slide_ppt/Picture1.6.png)  
![step3](./slide_ppt/Picture1.7.png)  
  
  
2.) Gesture estimation and object detection The video or image is given as input to the model. And frames are extracted from the video and sent for evaluation to extract the key points from all 33 key points for a single person.
Create a pose landmark from the prototype figure in which the coordinates (x, y, z) of the key point 33 skeleton landmarks are calculated. For all of these 33 objects, the object detector is shown in Figure 2.
![step4](./slide_ppt/Picture2.png)

Then the Gesture Estimator will cut off the human part from the entered image. Whereas the estimator takes a 256x256 resolution image of the person it detects as input, creates a real-time pose landmark, while checking for land mask similarity export key points. If the landmark visibility of the prototype and the landmark of the photographed person is less than or equal to 0.3, the blue key points is the correct pose. And if the landmark visibility of the prototype photo and the person's landmark, if it is greater than 0.3, the landmark (or key points) will be red. The pose is wrong at that point as shown in picture 3.
  
![step5](./slide_ppt/Picture3.1.png)
![step6](./slide_ppt/Picture3.2.png)
  
and running at over 30 frames per second on Pixel 2 phones. For effective detection, the speedy Tiny YOLO V3 algorithm is used. and high humility Let's detect the person who created the frame to suggest the pose location, the final step set, when the pose counts the landmark's wrong points less than or equal to 5 points, it will automatically take a photo and keep this picture.

**Result :**  
  
![step6](./slide_ppt/Picture4.png)

## About Program  
![Program_0](./slide_ppt/slide_No.0.png)  
![Program_1](./slide_ppt/slide_No.1.png)  
![Program_2](./slide_ppt/slide_No.2.png)  
![Program_3](./slide_ppt/slide_No.3.png)  
![Program_4](./slide_ppt/slide_No.4.png)  
![Program_5](./slide_ppt/slide_No.5.png)  
![Program_6](./slide_ppt/slide_No.6.png)  
![Program_7](./slide_ppt/slide_No.7.png)  
![Program_8](./slide_ppt/slide_No.8.png)  
![Program_9](./slide_ppt/slide_No.9.png)  
![Program_9](./slide_ppt/slide_No.10.png)  
