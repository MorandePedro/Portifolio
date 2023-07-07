import numpy as np

lines = np.random.random([20,20]).flatten()
with open('weights.txt', 'w') as f:
    for number in lines: f.write(f"{str(number)} ")
    