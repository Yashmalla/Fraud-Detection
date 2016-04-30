import numpy as np
matrix = np.random.random_integers(80, 100, (20,15))
print(matrix)
np.savetxt('ADAM.txt', matrix)
