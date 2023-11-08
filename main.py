import cv2
import numpy as np
from time import sleep

captura = cv2.VideoCapture("videos/traffic.mp4")
_, frame = captura.read()

subtracao = cv2.createBackgroundSubtractorMOG2(history=50, detectShadows=False, varThreshold=200)

#ROI
bbox = cv2.selectROI(frame, False)
(w1, h1, w2, h2) = bbox

#Linhas em relação ao ROI
linha1 = int(h1)
linha2 = int(h2 - 20)
linhas_color = (255, 255, 255)


def gerar_kernel(tipo_kernel):
    if tipo_kernel == "dilation":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    if tipo_kernel == "opening":
        kernel = np.ones((10, 10), np.uint8)
    if tipo_kernel == "closing":
        kernel = np.ones((10, 10), np.uint8)
    return kernel

def aplicar_filtro(subtraido):
    closing = cv2.morphologyEx(subtraido, cv2.MORPH_CLOSE, gerar_kernel("closing"), iterations=3)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, gerar_kernel("opening"), iterations=3)
    dilation = cv2.dilate(opening, gerar_kernel("dilation"), iterations=3)
    return dilation

def selecionar_area(frame):
  roi = frame[h1:h1 + h2, w1:w1 + w2]
  return roi

def detectar_veiculo(dilatado):
    contours, _ = cv2.findContours(dilatado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x + w1, y + h1), (x + w1 + w, y + h1 + h), (255, 255, 0), 2)

            centro_x = x + w1 + w // 2
            centro_y = y + h1 + h // 2
            cv2.circle(frame, (centro_x, centro_y), 2, (0, 255, 255), -1)

while True:
  ret, frame = captura.read()
  tempo = float(1/60)
  if frame is None:
    break 
  sleep(tempo) 

  roi = selecionar_area(frame)
  roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
  subtraido = subtracao.apply(roi)
  dilatado = aplicar_filtro(subtraido)

  cv2.imshow('Video c/ filtros', dilatado)
  cv2.imshow("ROI", roi)

  frame = cv2.line(frame, (w1, linha1), (w1 + w2, linha1), linhas_color, 2)

  deteccao = detectar_veiculo(dilatado)

  cv2.imshow("Frame", frame)

  if cv2.waitKey(1) == ord ('q'): #Pressionar q para fechar
    break
captura.release()
cv2.destroyAllWindows()



