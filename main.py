import cv2
import numpy as np
from time import sleep

class SubtrairFundo:
    def __init__(self):
        self.subtracao = cv2.createBackgroundSubtractorMOG2(history=80, detectShadows=False, varThreshold=200)

    def aplicar_subtracao(self, roi_blur):
        return self.subtracao.apply(roi_blur)

class SelecionarRegiaoInteresse:
    def __init__(self, frame):
        self.bbox = cv2.selectROI(frame, False)
        (self.w1, self.h1, self.w2, self.h2) = self.bbox

    def selecionar_area(self, frame):
        return frame[self.h1:self.h1 + self.h2, self.w1:self.w1 + self.w2]

    def desenhar_linha(self, frame): 
        centro_x = self.w1 + self.w2 // 2
        centro_y = self.h1 + self.h2 // 2
        cv2.line(frame, (self.w1, centro_y), (self.w1 + self.w2, centro_y), (255, 255, 255), 2)
        

class ContadorVeiculos:
    def __init__(self, pos_linha, offset):
        self.pos_linha = pos_linha
        self.offset = offset
        self.veiculos = 0
        self.texto_contagem = "Veiculos detectados: 0"
        self.posicoes_anteriores = set()

    def contar_veiculos(self, detec):
        for(x, y) in detec:
            if (self.pos_linha + self.offset) > y > (self.pos_linha - self.offset):
                self.veiculos += 1
                detec.remove((x, y))
                self.texto_contagem = "Veiculos detectados:" + str(self.veiculos)

class DetectarVeiculos:
    def __init__(self, w1, h1, w2, h2, contador_veiculos):
        self.w1 = w1
        self.h1 = h1
        self.w2 = w2
        self.h2 = h2
        self.contador_veiculos = contador_veiculos
        self.detec = []

    def detectar_veiculos(self, frame, morfologias):
        contours, _ = cv2.findContours(morfologias, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        total_area_roi = self.w2 * self.h2 
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            normalized_area = (w * h) / total_area_roi
            if normalized_area >= 0.06:
                cv2.rectangle(frame, (x + self.w1, y + self.h1), (x + self.w1 + w, y + self.h1 + h), (0, 255, 0), 2)
                centro_x = x + self.w1 + w // 2
                centro_y = y + self.h1 + h // 2
                centro = centro_x, centro_y
                self.detec.append(centro)
                cv2.circle(frame, (centro_x, centro_y), 2, (0, 255, 255), -1)
                self.contador_veiculos.contar_veiculos(self.detec)
            
        cv2.putText(frame, self.contador_veiculos.texto_contagem, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
class Video:
    def __init__(self, caminho_video):
        self.captura = cv2.VideoCapture(caminho_video)
        _, self.frame = self.captura.read()

        self.subtracao_fundo = SubtrairFundo()
        self.selecao_roi = SelecionarRegiaoInteresse(self.frame)
        self.contador_veiculos = ContadorVeiculos(self.selecao_roi.h1 + self.selecao_roi.h2 // 2, 10)
        self.deteccao_veiculos = DetectarVeiculos(self.selecao_roi.w1, self.selecao_roi.h1, self.selecao_roi.w2, self.selecao_roi.h2, self.contador_veiculos)

    def processar_video(self):
        while True:
            ret, self.frame = self.captura.read()
            tempo = float(1/60)
            sleep(tempo)
            if self.frame is None:
                print("Fim do Video")
                break

            roi = self.selecao_roi.selecionar_area(self.frame)
            roi_escala_cinza = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_blur = cv2.GaussianBlur(roi_escala_cinza, (3,3), 0)

            subtraido = self.subtracao_fundo.aplicar_subtracao(roi_blur)
            
            morfologias = self.transformacao_morfologica(subtraido)

            self.selecao_roi.desenhar_linha(self.frame)

            self.deteccao_veiculos.detectar_veiculos(self.frame, morfologias)

            cv2.imshow('Morfologias', morfologias)
            cv2.imshow("ROI", roi)
            cv2.imshow("Resultado", self.frame)

            if cv2.waitKey(1) == ord('q'):  # Pressionar q para fechar
                break

        self.captura.release()
        cv2.destroyAllWindows()

    def transformacao_morfologica(self, subtraido):
        kernel = np.ones((5,5),np.uint8)
        closing = cv2.morphologyEx(subtraido, cv2.MORPH_CLOSE, kernel , iterations=5)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=2)
        dilated = cv2.dilate(opening, kernel, iterations=3)
        return dilated

video = Video("videos/traffic.mp4")
video.processar_video()