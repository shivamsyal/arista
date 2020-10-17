# CREDIT TO atulapra FOR EMOTION DETECTION
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import pyttsx3 as p
from os import system
import os
import subprocess
from tkinter import *
import imghdr as i
import smtplib
tk=Tk()
from PIL import Image
from PIL import ImageTk
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

train_dir = 'data/train'
val_dir = 'data/test'

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 50

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)


# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

file1=open("engagement_level.txt","w")
file1.write("0,0\n")
file1.close()

# print("Installing all required modules")
# os.system("pip install -r requirements.txt")
file = open("engagement_level.txt","r+")
e=p.init()
face_csc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cam = cv2.VideoCapture(0)
i=0
start=0
a,b,c,d=0,0,0,0



def start_pos():
    pos_val=0
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "model.caffemodel")
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    fac=0
    a,b=0,0
    k=0
    while True:
        command="brightness 0.1"
        frame = vs.read()
        
        frame = imutils.resize(frame, width=400)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        k+=1
        
        for i in range(0, detections.shape[2]):
            
                confidence = detections[0, 0, i, 2]
                if confidence < 0.7:
                        continue
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                if endX-startX>75:
                        file.write(str(k)+","+str(endX-startX)+"\n")
                        a+=1
                        if a==1:
                                system("say Please move back")
                        os.system(command)
                elif endX-startX<50:
                        file.write(str(k)+","+str(endX-startX)+"\n")
                        b+=1
                        if b==1:
                                system("say Please move forward")
                        os.system(command)
                else:
                        file.write(str(k)+","+str(endX-startX)+"\n")
                        os.system("brightness 0.85")
                        a, b=0,0
                
                text = "{:.2f}%".format(confidence * 100)
                pos_val+=(endX-startX)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                
                
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.imshow("Posture Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print(pos_val/k)
            file.close()
            stop_pos()
            cv2.destroyAllWindows()
            vs.stop()
            break
def stop_pos():
    file.close()
    cam.release()
    cv2.destroyAllWindows()
def start_emotion():
    # emotions will be displayed on your face from the webcam feed
    model.load_weights('model.h5')
    cv2.ocl.setUseOpenCL(False)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Emotion Detection', cv2.resize(frame,(1000,600),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



tk.title("Arista Engagement System")
canvas=Canvas(width=1120,height=630,bg="white")
canvas.pack()
photo=PhotoImage(file="arista_logo.png")
photo=photo.subsample(20)
panel = Label(tk, image= photo)
canvas.create_image(560,40, anchor=N, image=photo)

canvas.create_text(250,150,text="Start your posture session:")
canvas.create_text(250,300,text="Start your emotion capture:")
choose=Button(text="Start",command=start_emotion).place(x=230, y=340)

choose=Button(text="Start",command=start_pos).place(x=230, y=190)
canvas.create_text(560,150,text="OR")
canvas.create_text(900,150,text="Stop your posture session:")

canvas.create_text(900,200,text="To stop session, press Ctrl+C")
cam.release()
cv2.destroyAllWindows()
tk.mainloop()
