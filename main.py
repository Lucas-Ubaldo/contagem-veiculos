import cv2
import numpy as np
from time import sleep

#Recebe o vídeo
captura = cv2.VideoCapture("traffic.mp4")
subtracao = cv2.createBackgroundSubtractorMOG2(detectShadows=True)


while True:
  ret, frame = captura.read() 
  tempo = float(1/60)
  sleep(tempo) 

  escala_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
  vid_subtraido = subtracao.apply(frame)
  real_part = cv2.bitwise_and(frame, frame, mask=vid_subtraido)
  vid_sub3 = cv2.cvtColor(vid_subtraido, cv2.COLOR_GRAY2BGR)

  dilat = cv2.dilate(vid_sub3, np.ones((5,5)))
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
  dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
  dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

  cv2.imshow('Video original', frame)
  #cv2.imshow('Video em escala de cinza', escala_cinza)
  cv2.imshow("Video Subtraido", vid_sub3)
  #cv2.imshow("Filtro dilatação", dilatada)

  if cv2.waitKey(1) == ord ('q'):
    break
captura.release()
cv2.destroyAllWindows()


