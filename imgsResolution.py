from PIL import Image
import IPython
import numpy as np

img = Image.open("download.jpeg")

array = np.array(img)
# len(array[0:int(len(array)/(len(array)/20))])
# pegando o intervalo necessário para 
# fazer a soma e tirar a média e converter para 1px

converted_rows = []
converted_columns = []
indices = np.array(range(0,len(array),int(len(array)/(len(array)/20))))
init = 0
final = 1
for ind in indices:
    for intervalo in array[ind[init]:ind[final]]:
        gray_scale=0
        for pixel in intervalo:
            R = pixel[0] / 3
            G = pixel[1] / 3
            B = pixel[2] / 3
            if (R+G+B) == 0:
                gray_scale += 1.0
            else:
                gray_scale += 1/(R+G+B)
        converted_rows.append(gray_scale/int(len(array)/(len(array)/20)))
    init += 1
    final += 1
IPython.embed()
