import cv2
import numpy
from time import sleep

#Recebe o vídeo
captura = cv2.VideoCapture("traffic.mp4")

while True:
  _, frame = captura.read() #faz a leitura de cada frame do vídeo
  tempo = float(1/60)
  sleep(tempo) #Necessário para "normalizar" a velocidade do vídeo
  escala_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  cv2.imshow('Video original', frame)
  cv2.imshow('Video em escala de cinza', escala_cinza)
  if cv2.waitKey(1) == ord ('q'):
    break
captura.release()
cv2.destroyAllWindows()


