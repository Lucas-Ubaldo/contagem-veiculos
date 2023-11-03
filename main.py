import cv2
import numpy as np
from time import sleep

captura = cv2.VideoCapture("videos/highway2.mp4")
hasFrame, frame = captura.read()

subtracao = cv2.createBackgroundSubtractorMOG2(history=50, detectShadows=False, varThreshold=200)

#ROI
bbox = cv2.selectROI(frame, False)
(w1, h1, w2, h2) = bbox

#Linhas em relação ao ROI
linha1 = int(h1)
linha2 = int(h2 - 20)
linhas_color = (255, 255, 255)


def getKernel(tipo_kernel):
    if tipo_kernel == "dilation":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    if tipo_kernel == "opening":
        kernel = np.ones((5, 5), np.uint8)
    if tipo_kernel == "closing":
        kernel = np.ones((10, 10), np.uint8)
    return kernel

def getFilter(subtraido):
    closing = cv2.morphologyEx(subtraido, cv2.MORPH_CLOSE, getKernel("closing"), iterations=2)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, getKernel("opening"), iterations=2)
    dilation = cv2.dilate(opening, getKernel("dilation"), iterations=2)
    return dilation

def getRoi(frame):
  roi = frame[h1:h1 + h2, w1:w1 + w2]
  return roi

while True:
  ret, frame = captura.read()
  tempo = float(1/60)
  if frame is None:
    break 
  sleep(tempo) 

  escala_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  roi = getRoi(frame)
    
  subtraido = subtracao.apply(roi)
  dilatado = getFilter(subtraido)

  cv2.imshow('Video c/ filtros', dilatado)
  cv2.imshow("ROI", roi)

  frame = cv2.line(frame, (w1, linha1), (w1 + w2, linha1), linhas_color, 2)
  frame = cv2.line(frame, (w1, h1 + linha2), (w1 + w2, h1 + linha2), linhas_color, 2)
  cv2.imshow("Frame", frame)

  if cv2.waitKey(1) == ord ('q'): #Pressionar q para fechar
    break
captura.release()
cv2.destroyAllWindows()



