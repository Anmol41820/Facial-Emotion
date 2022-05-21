from tkinter import*
from tkinter import ttk
# from tkinter.font import BOLD
# from turtle import st
from PIL import Image , ImageTk
from tkinter import messagebox

from scipy.misc import face
import mysql.connector 
import cv2
from deepface import DeepFace



class Face_Emotion:
    def __init__(self,root):
        self.root=root
        self.root.geometry("1530x790+0+0")  #geometry of the window
        self.root.title("face Recognition System")


        faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        cap=cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannt open webcam")

        while True:
            ret,frame = cap.read()

            result = DeepFace.analyze(frame,actions=['emotion'])
            # result1 = DeepFace.analyze(frame,actions=['age'])

            gray=cv2.cvtColor(frame,cv2.COLOR_BGRA2GRAY)
            faces = faceCascade.detectMultiScale(gray,1.1,4)

            for(x,y,w,h) in faces:
                cv2.rectangle(frame ,(x,y),(x+w,y+h), (0,255,0), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(frame,result['dominant_emotion'],(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)

            # cv2.putText(frame,result1['age'],(100,40),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)

            # cv2.putText(frame,result['gender'],(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)

            # cv2.putText(frame,result['race'],(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)

            cv2.imshow('Face Emotion Fun Play',frame)

            if cv2.waitKey(2) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

#==========calling main============
if __name__=="__main__":
    root=Tk()
    obj=Face_Emotion(root)
    root.mainloop()