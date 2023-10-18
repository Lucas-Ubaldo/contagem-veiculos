import cv2
import numpy as np
from time import sleep

captura = cv2.VideoCapture("videos/highway2.mp4")
hasFrame, frame = captura.read()
subtracao = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

bbox = cv2.selectROI(frame, False)
(w1, h1, w2, h2) = bbox

def main():
  tempo = float(1/60)
  
  while True:
    ret, frame = captura.read() 
    sleep(tempo) 

    roi = frame[h1:h1 + h2, w1:w1 + w2]
    
    escala_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    vid_subtraido = subtracao.apply(roi)

    cv2.imshow('Video original', frame)
    cv2.imshow("video", vid_subtraido)

    if cv2.waitKey(1) == ord ('q'):
      break
  captura.release()
  cv2.destroyAllWindows()
main()


