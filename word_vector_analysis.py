import pandas as pd
import numpy as np

print('Loading Word Vector Array...')
data = np.load('word_vectors.npy')
print('Loaded.')
print(data)

for d in list(data[:,0]):
    print(d)

