import numpy as np
import pandas as pd
import IPython
import os
from PIL import Image

class RedeNeural:
    def __init__(self, learning_rate, pesos, bias):
        self.pesos = pesos
        self.bias = bias
        self.learning_rate = learning_rate
    
    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))
    
    def _previsao(self,input_array):
        camada_1 = np.dot(input_array, self.pesos) + self.bias
        camada_2 = self._sigmoid(camada_1)
        
        if camada_2 < 0.30:
            return 1
        elif 0.30 < camada_2 < 0.50:
            return 2
        elif 0.50 < camada_2 < 0.70:
            return 3
        elif 0.70 < camada_2 < 0.90:
            return 4
        elif camada_2 > 0.90:
            return 5

    def _calcula_derivadas(self, input_array, alvo):
        camada_1 = np.dot(input_array, self.pesos) + self.bias
        camada_2 = self._sigmoid(camada_1)
        previsao = camada_2

        derro_dprevisao = 2 * (previsao - alvo)
        dprevisao_camada1 = self._sigmoid_deriv(camada_1)
        dbias_camada1 = 1
        dpesos_camada1 = (0 * self.pesos) + (1 + input_array)

        derro_bias = (
            derro_dprevisao * dprevisao_camada1 * dbias_camada1
        )

        derro_pesos = (
            derro_dprevisao * dprevisao_camada1 * dpesos_camada1
        )
        return derro_bias, derro_pesos
    
    def _atualiza_params(self, derro_bias, derro_pesos):
        self.bias = self.bias - (derro_bias * self.learning_rate)
        self.pesos = self.pesos - (derro_pesos * self.learning_rate)
    
    def _treinar(self,array_inputs, alvos, interacoes):
        erros = []
        for interacao in range(interacoes):
            rand_index = np.random.randint(len(array_inputs))

            input_array = array_inputs[rand_index]
            input_target = alvos[rand_index]

            erro_bias, erro_pesos = self._calcula_derivadas(input_array,input_target)
            self._atualiza_params(erro_bias, erro_pesos)

            if interacao % 100 == 0:
                error = 0
                for data_instace_index in range(len(array_inputs)):
                    array = array_inputs[data_instace_index]
                    alvo = alvos[data_instace_index]

                    previsao = self._previsao(array)
                    erro = np.square(previsao - alvo)

                    error += erro
                erros.append(error)
        return erros, self.pesos, self.bias


def img2GrayScaleMatrix(img_file):
    img = Image.open(img_file)
    img = img.resize((20, 20), Image.LANCZOS)
    img_array = np.asarray(img)
    gray_array = []

    for row in img_array:
        gray_row = []
        for pixel in row:
            R = pixel[0] / 3
            G = pixel[1] / 3
            B = pixel[2] / 3
            if (R+G+B) == 0:
                gray_scale = 1.0
            else:
                gray_scale = 1/(R+G+B)

            if gray_scale < 0.1:
                gray_row.append(0)
            else:
                gray_row.append(gray_scale)
        gray_array.append(gray_row)
    return np.array(gray_array).flatten()

# rede_neural = RedeNeural(0.1)
# rede_neural._treinar(arrays,alvos,100000)
