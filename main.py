import cv2
import numpy as np
from time import sleep

class SubtrairFundo:
    def __init__(self):
        self.subtracao = cv2.createBackgroundSubtractorMOG2(history=25, detectShadows=False, varThreshold=100)

    def aplicar_subtracao(self, frame):
        return self.subtracao.apply(frame)

class SelecionarRegiaoInteresse:
    def __init__(self, frame):
        self.bbox = cv2.selectROI(frame, False)
        (self.w1, self.h1, self.w2, self.h2) = self.bbox

    def selecionar_area(self, frame):
        return frame[self.h1:self.h1 + self.h2, self.w1:self.w1 + self.w2]

    def desenhar_linhas(self, frame):
        cv2.line(frame, (self.w1, self.h1), (self.w1 + self.w2, self.h1), (255, 255, 255), 2)
        cv2.line(frame, (self.w1, self.h1 + self.h2), (self.w1 + self.w2, self.h1 + self.h2), (255, 255, 255), 2)

class ContadorVeiculos:
    def __init__(self, pos_linha, offset):
        self.pos_linha = pos_linha
        self.offset = offset
        self.carros = 0
        self.texto_contagem = "Carros detectados: 0"
        self.posicoes_anteriores = set()

    def contar_veiculos(self, centro_y):
        if (self.pos_linha - self.offset) < centro_y < (self.pos_linha + self.offset):
            if centro_y not in self.posicoes_anteriores:
                self.carros += 1
                self.texto_contagem = "Carros detectados: " + str(self.carros)
                self.posicoes_anteriores.add(centro_y)

class DetectarVeiculo:
    def __init__(self, w1, h1, w2, contador_veiculos):
        self.w1 = w1
        self.h1 = h1
        self.w2 = w2
        self.contador_veiculos = contador_veiculos

    def detectar_veiculo(self, frame, dilatado):
        contours, _ = cv2.findContours(dilatado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)

                cv2.rectangle(frame, (x + self.w1, y + self.h1), (x + self.w1 + w, y + self.h1 + h), (0, 255, 0), 2)
                centro_x = x + self.w1 + w // 2
                centro_y = y + self.h1 + h // 2
                cv2.circle(frame, (centro_x, centro_y), 2, (0, 255, 255), -1)

                self.contador_veiculos.contar_veiculos(centro_y)

        
        cv2.putText(frame, self.contador_veiculos.texto_contagem, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

class Video:
    def __init__(self, caminho_video):
        self.captura = cv2.VideoCapture(caminho_video)
        _, self.frame = self.captura.read()

        self.subtracao_fundo = SubtrairFundo()
        self.selecao_roi = SelecionarRegiaoInteresse(self.frame)
        self.contador_veiculos = ContadorVeiculos(self.selecao_roi.h1 + self.selecao_roi.h2 // 2, 10)
        self.deteccao_veiculo = DetectarVeiculo(self.selecao_roi.w1, self.selecao_roi.h1, self.selecao_roi.w2, self.contador_veiculos)

    def processar_video(self):
        while True:
            ret, self.frame = self.captura.read()
            tempo = float(1/60)
            if self.frame is None:
                break
            sleep(tempo)

            roi = self.selecao_roi.selecionar_area(self.frame)
            roi_tons_cinza = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            subtraido = self.subtracao_fundo.aplicar_subtracao(roi_tons_cinza)
            dilatado = self.aplicar_filtros(subtraido)

            cv2.imshow('Video c/ filtros', dilatado)
            cv2.imshow("ROI", roi)

            self.selecao_roi.desenhar_linhas(self.frame)

            self.deteccao_veiculo.detectar_veiculo(self.frame, dilatado)

            cv2.imshow("Frame", self.frame)

            if cv2.waitKey(1) == ord('q'):  # Pressionar q para fechar
                break

        self.captura.release()
        cv2.destroyAllWindows()

    def aplicar_filtros(self, subtraido):
        closing = cv2.morphologyEx(subtraido, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8), iterations=3)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, np.ones((10, 10), np.uint8), iterations=3)
        dilated = cv2.dilate(opening, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)), iterations=3)
        return dilated

video = Video("videos/traffic.mp4")
video.processar_video()
