from flask import Flask, request, send_file
from NN_numbers_POO import RedeNeural, img2GrayScaleMatrix
import os
import IPython
import numpy as np

weights_file = open("weights.txt").readlines()[0].split(' ')
weights_file.pop()
weights = []
for w in weights_file: 
    weights.append(float(w))
weights = np.array(weights)

bias = float(open("bias.txt").readlines()[0])

folder1 = 'py_imgs\\1'
folder2 = 'py_imgs\\2'
folder3 = 'py_imgs\\3'
folder4 = 'py_imgs\\4'
folder5 = 'py_imgs\\5'

folders = [folder1,folder2,folder3,folder4,folder5]

arrays = []
alvos = []
numeros = {}
for names in folders:
    number = names.split('\\')[len(names.split('\\'))-1]
    numeros[number] = []
    for file in os.listdir(names):
        arrays.append(img2GrayScaleMatrix(f'{names}\\{file}'))
        numeros[number].append(img2GrayScaleMatrix(f'{names}\\{file}'))
        alvos.append(int(number))
alvos = np.array(alvos)/5
arrays = np.array(arrays)
IPython.embed()
rede_neural = RedeNeural(0.1, weights, bias)
erros, new_pesos, new_bias = rede_neural._treinar(arrays,alvos,100000)



teste1 = img2GrayScaleMatrix('py_imgs\\5\\download.jpeg')

RedeNeural = RedeNeural(0.1, weights, bias)
IPython.embed()

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['file']
    file.save('py_imgs/img.jpg')

    return 'Imagem recebida e processada com sucesso.', 200

@app.route('/imagem_processada', methods=['GET'])
def obter_imagem_processada():
    caminho_imagem_processada = 'py_imgs/img.jpg'

    # Envie a imagem processada como resposta para o front end
    return send_file(caminho_imagem_processada, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run()